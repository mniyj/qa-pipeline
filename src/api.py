import sys
import json
import shutil
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load .env before anything else so API keys are in os.environ
load_dotenv(Path(__file__).parent.parent / ".env")

# orchestrator.py uses bare imports (from doc_preprocessor import ...);
# add src/ to sys.path so those resolve when called from project root.
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, APIRouter, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="Insurance QA Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "data" / "raw_documents"
ALLOWED_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md"}

# ─── Documents router ────────────────────────────────────────────────────────

router_docs = APIRouter(prefix="/api/documents", tags=["documents"])


@router_docs.post("/upload")
async def upload_document(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"不支持的格式: {ext}")
    dest = DOCS_DIR / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "size": dest.stat().st_size}


@router_docs.get("/")
async def list_documents(page: int = 1, size: int = 50):
    all_files = []
    if DOCS_DIR.exists():
        for path in sorted(DOCS_DIR.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                all_files.append({
                    "name": path.name,
                    "relative_path": str(path.relative_to(DOCS_DIR)),
                    "size": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                })
    total = len(all_files)
    size = min(max(size, 1), 200)
    skip = (page - 1) * size
    items = all_files[skip:skip + size]
    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "pages": max(1, (total + size - 1) // size),
    }


@router_docs.delete("/{filename:path}")
async def delete_document(filename: str):
    path = DOCS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "文件不存在")
    path.unlink()
    return {"deleted": filename}


# ─── Pipeline router ─────────────────────────────────────────────────────────

router_pipeline = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router_pipeline.get("/status")
async def get_pipeline_status():
    from src.pipeline_state import pipeline_state
    return pipeline_state.to_dict()


@router_pipeline.post("/start")
async def start_pipeline(body: dict, background_tasks: BackgroundTasks):
    from src.pipeline_state import pipeline_state
    from src.pipeline_runner import run_step
    step = body.get("step", "all")
    if step not in ("preprocess", "seed", "seed_dedup", "expand", "qc", "all"):
        raise HTTPException(400, f"未知步骤: {step}")
    if pipeline_state.is_running:
        raise HTTPException(409, "流水线正在运行中")
    background_tasks.add_task(run_step, step)
    return {"started": step}


@router_pipeline.post("/stop")
async def stop_pipeline():
    from src.pipeline_state import pipeline_state
    if not pipeline_state.is_running:
        raise HTTPException(409, "流水线未在运行")
    pipeline_state.request_stop()
    return {"stopped": True}


@router_pipeline.get("/events")
async def pipeline_events(request: Request):
    from src.pipeline_state import pipeline_state
    pipeline_state.reset_queue()

    async def generator():
        yield {"data": json.dumps({"type": "init", "state": pipeline_state.to_dict()})}
        q = pipeline_state._get_queue()
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
                yield {"data": json.dumps(event)}
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"type": "ping"})}

    return EventSourceResponse(generator())


# ─── Data router ─────────────────────────────────────────────────────────────

router_data = APIRouter(prefix="/api/data", tags=["data"])


@router_data.get("/chunks/stats")
async def chunks_stats():
    """实时统计 chunks.jsonl 中的文档数、chunk 总数及每份文档的分布"""
    def _calc():
        from collections import Counter
        filepath = BASE_DIR / "data" / "chunks" / "chunks.jsonl"
        if not filepath.exists():
            return {"total_chunks": 0, "total_docs": 0, "min": 0, "max": 0, "median": 0}
        per_doc: Counter = Counter()
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    per_doc[json.loads(line).get("doc_file", "")] += 1
        counts = sorted(per_doc.values())
        return {
            "total_chunks": sum(counts),
            "total_docs":   len(counts),
            "min":          counts[0]  if counts else 0,
            "max":          counts[-1] if counts else 0,
            "median":       counts[len(counts) // 2] if counts else 0,
        }
    return await asyncio.to_thread(_calc)


@router_data.get("/chunks")
async def list_chunks(
    page: int = 1,
    size: int = 20,
    doc_type: str = "",
    insurance_type: str = "",
    search: str = "",
):
    from src.data_reader import read_jsonl_paged
    filepath = BASE_DIR / "data" / "chunks" / "chunks.jsonl"
    filters = {k: v for k, v in {"doc_type": doc_type, "insurance_type": insurance_type}.items() if v}
    return await asyncio.to_thread(read_jsonl_paged, filepath, page, size, filters, search)


@router_data.get("/qa/filters")
async def qa_filter_options():
    from src.data_reader import get_distinct_values
    filepath = BASE_DIR / "data" / "qc_results" / "qa_passed.jsonl"
    if not filepath.exists():
        filepath = BASE_DIR / "data" / "final" / "qa_final.jsonl"

    def _load():
        return {
            "insurance_types":    get_distinct_values(filepath, "insurance_type"),
            "difficulties":       get_distinct_values(filepath, "difficulty"),
            "question_types":     get_distinct_values(filepath, "question_type"),
            "generation_methods": get_distinct_values(filepath, "generation_method"),
        }
    return await asyncio.to_thread(_load)


@router_data.get("/qa")
async def list_qa(
    page: int = 1,
    size: int = 20,
    status: str = "passed",
    insurance_type: str = "",
    difficulty: str = "",
    question_type: str = "",
    generation_method: str = "",
    search: str = "",
):
    from src.data_reader import read_jsonl_paged
    file_map = {
        "passed":   BASE_DIR / "data" / "qc_results" / "qa_passed.jsonl",
        "rejected": BASE_DIR / "data" / "qc_results" / "qa_rejected.jsonl",
        "final":    BASE_DIR / "data" / "final" / "qa_final.jsonl",
    }
    filepath = file_map.get(status, file_map["passed"])
    filters = {k: v for k, v in {
        "insurance_type":    insurance_type,
        "difficulty":        difficulty,
        "question_type":     question_type,
        "generation_method": generation_method,
    }.items() if v}
    return await asyncio.to_thread(read_jsonl_paged, filepath, page, size, filters, search)


@router_data.get("/reports/overview")
async def reports_overview():
    def _load():
        def read_json(path: Path) -> dict:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

        pre_report = read_json(BASE_DIR / "data" / "chunks" / "preprocess_report.json")
        qc_report  = read_json(BASE_DIR / "data" / "qc_results" / "qc_report.json")

        seeds_dir    = BASE_DIR / "data" / "seeds"
        expanded_dir = BASE_DIR / "data" / "expanded"

        seeds_count = sum(
            sum(1 for line in open(f, encoding="utf-8") if line.strip())
            for f in seeds_dir.glob("seed_*.jsonl")
            if f.exists()
        ) if seeds_dir.exists() else 0

        filter_file = expanded_dir / "seed_dedup_filter.json"
        dup_count = len(json.loads(filter_file.read_text(encoding="utf-8"))) if filter_file.exists() else 0

        expanded_count = sum(
            sum(1 for line in open(f, encoding="utf-8") if line.strip())
            for f in expanded_dir.glob("*.jsonl")
            if f.exists() and not f.name.endswith("_report.json")
        ) if expanded_dir.exists() else 0

        return {
            "preprocessing":       pre_report,
            "seeds_total":         seeds_count,
            "seeds_dedup_removed": dup_count,
            "seeds_after_dedup":   seeds_count - dup_count,
            "expanded_total":      expanded_count,
            "qc":                  qc_report,
        }
    return await asyncio.to_thread(_load)


# ─── Register routers & mount frontend ───────────────────────────────────────

app.include_router(router_docs)
app.include_router(router_pipeline)
app.include_router(router_data)

frontend_dir = BASE_DIR / "frontend"
frontend_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")
