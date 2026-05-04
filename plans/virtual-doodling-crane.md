# Insurance QA Pipeline Frontend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a web UI for uploading documents, controlling the 4-step QA generation pipeline with real-time SSE progress, and browsing/filtering/paginating the generated Q&A pairs.

**Architecture:** FastAPI backend wraps the existing `cmd_*` functions in `orchestrator.py`; pipeline steps run in a background thread so they don't block the event loop; a custom logging handler captures all `logger.info()` calls from pipeline modules and forwards them to an `asyncio.Queue`, which an SSE endpoint drains for real-time progress. The frontend is a single `frontend/index.html` using Vue 3 CDN + Tailwind CDN + Chart.js CDN — no Node/npm build step.

**Tech Stack:** FastAPI 0.115+, uvicorn[standard], python-multipart, sse-starlette; Vue 3 CDN; Tailwind CSS CDN; Chart.js CDN

**Critical files to modify/create:**
- Create: `src/api.py` — FastAPI app
- Create: `src/pipeline_state.py` — shared progress state + SSE logging handler
- Create: `src/data_reader.py` — server-side JSONL pagination
- Modify: `requirements.txt` — add FastAPI deps
- Create: `frontend/index.html` — complete SPA

**Import path note:** `orchestrator.py` uses bare imports (`from doc_preprocessor import ...`). `api.py` must add `src/` to `sys.path` before importing orchestrator functions.

---

## Task 1: Install Dependencies & Minimal Server

**Files:**
- Modify: `requirements.txt`
- Create: `src/api.py`
- Create: `frontend/index.html` (scaffold)

**Step 1: Add to requirements.txt**

Append these lines:
```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
sse-starlette>=2.1.0
```

**Step 2: Create `src/api.py`**

```python
import sys
from pathlib import Path

# orchestrator uses bare imports; add src/ to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Insurance QA Pipeline")

# Register routers (added in later tasks)
# app.include_router(router_docs)
# app.include_router(router_pipeline)
# app.include_router(router_data)

frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")
```

**Step 3: Create `frontend/index.html` scaffold**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>Insurance QA Pipeline</title>
</head>
<body>
  <h1>Insurance QA Pipeline</h1>
  <p>Frontend coming soon…</p>
</body>
</html>
```

**Step 4: Install deps and verify server starts**

```bash
pip install fastapi "uvicorn[standard]" python-multipart sse-starlette
uvicorn src.api:app --reload --port 8000
# visit http://localhost:8000 — should show "Insurance QA Pipeline"
```

**Step 5: Commit**

```bash
git add requirements.txt src/api.py frontend/index.html
git commit -m "feat: scaffold FastAPI server with static file serving"
```

---

## Task 2: Pipeline State Manager + SSE Log Handler

**Files:**
- Create: `src/pipeline_state.py`

This is the core of real-time progress. A singleton `PipelineState` object tracks each step's status. A `PipelineLogHandler` subclasses `logging.Handler` and forwards every `logger.info/warning/error` emitted by pipeline modules into the asyncio event queue — no changes needed to `doc_preprocessor.py`, `seed_generator.py`, etc.

**Step 1: Create `src/pipeline_state.py`**

```python
import asyncio
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StepStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class StepState:
    key: str
    label: str
    status: StepStatus = StepStatus.IDLE
    progress: int = 0
    message: str = ""
    output_count: int = 0
    error: str = ""


class PipelineState:
    def __init__(self):
        self._lock = threading.Lock()
        self.steps: dict[str, StepState] = {
            "preprocess": StepState("preprocess", "文档预处理"),
            "seed":       StepState("seed",       "种子生成"),
            "expand":     StepState("expand",     "批量扩展"),
            "qc":         StepState("qc",         "质量检查"),
        }
        self.current_step: Optional[str] = None
        self.is_running: bool = False
        self._queue: Optional[asyncio.Queue] = None

    def _get_queue(self) -> Optional[asyncio.Queue]:
        """Lazily return the queue once an event loop is running."""
        if self._queue is None:
            try:
                loop = asyncio.get_running_loop()
                self._queue = asyncio.Queue()
            except RuntimeError:
                pass
        return self._queue

    def reset_queue(self):
        """Call at the start of a pipeline run to flush stale events."""
        self._queue = asyncio.Queue()

    def update_step(self, step: str, **kwargs):
        with self._lock:
            s = self.steps[step]
            for k, v in kwargs.items():
                setattr(s, k, v)
        self._emit({"type": "step_update", "step": step, **kwargs})

    def log(self, message: str, level: str = "INFO"):
        self._emit({"type": "log", "message": message, "level": level})

    def _emit(self, event: dict):
        q = self._get_queue()
        if q is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(q.put_nowait, event)
        except (RuntimeError, asyncio.QueueFull):
            pass

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "current_step": self.current_step,
                "is_running": self.is_running,
                "steps": {
                    k: {
                        "key": v.key,
                        "label": v.label,
                        "status": v.status,
                        "progress": v.progress,
                        "message": v.message,
                        "output_count": v.output_count,
                        "error": v.error,
                    }
                    for k, v in self.steps.items()
                },
            }


pipeline_state = PipelineState()


class PipelineLogHandler(logging.Handler):
    """Attaches to the root logger; forwards records to SSE queue."""

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        pipeline_state.log(msg, record.levelname)
```

**Step 2: Commit**

```bash
git add src/pipeline_state.py
git commit -m "feat: add pipeline state manager and SSE log handler"
```

---

## Task 3: Document Upload & List Endpoints

**Files:**
- Modify: `src/api.py`

**Step 1: Add router in `src/api.py`**

Insert before the `app.mount(...)` line:

```python
import shutil
from fastapi import APIRouter, UploadFile, HTTPException

DOCS_DIR = Path(__file__).parent.parent / "data" / "raw_documents"
ALLOWED_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md"}

router_docs = APIRouter(prefix="/api/documents", tags=["documents"])

@router_docs.post("/upload")
async def upload_document(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"不支持的格式: {ext}")
    dest = DOCS_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "size": dest.stat().st_size}

@router_docs.get("/")
async def list_documents():
    files = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            files.append({
                "name": path.name,
                "relative_path": str(path.relative_to(DOCS_DIR)),
                "size": path.stat().st_size,
                "modified": path.stat().st_mtime,
            })
    return {"files": files, "total": len(files)}

@router_docs.delete("/{filename:path}")
async def delete_document(filename: str):
    path = DOCS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "文件不存在")
    path.unlink()
    return {"deleted": filename}
```

Uncomment `app.include_router(router_docs)`.

**Step 2: Test**

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@data/raw_documents/laws/01_中华人民共和国保险法.md"
# Expected: {"filename": "...", "size": N}

curl http://localhost:8000/api/documents/
# Expected: {"files": [...], "total": N}
```

**Step 3: Commit**

```bash
git commit -m "feat: add document upload, list, delete endpoints"
```

---

## Task 4: Pipeline Runner (Background Thread)

**Files:**
- Create: `src/pipeline_runner.py`
- Modify: `src/api.py`

The runner executes `cmd_*` functions from `orchestrator.py` in a background thread. It injects `PipelineLogHandler` into the root logger for the duration of the run so all `logger.info()` calls in all pipeline modules become SSE events.

**Step 1: Create `src/pipeline_runner.py`**

```python
import logging
import threading
from src.pipeline_state import pipeline_state, PipelineLogHandler

# Import orchestrator cmd functions (src/ is already on sys.path via api.py)
from orchestrator import (
    load_config,
    cmd_preprocess,
    cmd_seed,
    cmd_expand,
    cmd_qc,
)


_run_lock = threading.Lock()


def run_step(step: str):
    """
    Called by FastAPI BackgroundTasks. Runs in a worker thread.
    step: "preprocess" | "seed" | "expand" | "qc" | "all"
    """
    if not _run_lock.acquire(blocking=False):
        return  # already running

    pipeline_state.reset_queue()

    # Attach log handler to root logger
    handler = PipelineLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    steps = ["preprocess", "seed", "expand", "qc"] if step == "all" else [step]

    try:
        pipeline_state.is_running = True
        config = load_config()

        for s in steps:
            pipeline_state.current_step = s
            pipeline_state.update_step(s, status="running", progress=0, error="")

            try:
                if s == "preprocess":
                    report = cmd_preprocess(config)
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("total_chunks", 0) if report else 0,
                    )
                elif s == "seed":
                    report = cmd_seed(config)
                    count = report.get("total_generated", 0) if report else 0
                    pipeline_state.update_step(s, status="done", progress=100, output_count=count)
                elif s == "expand":
                    cmd_expand(config)
                    pipeline_state.update_step(s, status="done", progress=100)
                elif s == "qc":
                    report = cmd_qc(config)
                    count = report.get("passed_total", 0) if report else 0
                    pipeline_state.update_step(s, status="done", progress=100, output_count=count)
            except Exception as e:
                pipeline_state.update_step(s, status="error", error=str(e))
                raise

    finally:
        root_logger.removeHandler(handler)
        pipeline_state.is_running = False
        pipeline_state.current_step = None
        _run_lock.release()
```

**Step 2: Add pipeline endpoints to `src/api.py`**

```python
import asyncio, json
from fastapi import BackgroundTasks, Request
from sse_starlette.sse import EventSourceResponse
from src.pipeline_state import pipeline_state
from src.pipeline_runner import run_step

router_pipeline = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

@router_pipeline.get("/status")
async def get_status():
    return pipeline_state.to_dict()

@router_pipeline.post("/start")
async def start_pipeline(body: dict, background_tasks: BackgroundTasks):
    step = body.get("step", "all")
    if step not in ("preprocess", "seed", "expand", "qc", "all"):
        raise HTTPException(400, f"Unknown step: {step}")
    if pipeline_state.is_running:
        raise HTTPException(409, "Pipeline already running")
    background_tasks.add_task(run_step, step)
    return {"started": step}

@router_pipeline.get("/events")
async def pipeline_events(request: Request):
    async def generator():
        # Send current state snapshot immediately
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
```

Uncomment `app.include_router(router_pipeline)`.

**Step 3: Test SSE**

```bash
# Terminal 1
curl -N http://localhost:8000/api/pipeline/events
# Should stream: data: {"type":"init","state":{...}}

# Terminal 2
curl -X POST http://localhost:8000/api/pipeline/start \
  -H "Content-Type: application/json" -d '{"step":"preprocess"}'
# Should see log events in Terminal 1
```

**Step 4: Commit**

```bash
git commit -m "feat: add pipeline runner with SSE progress streaming"
```

---

## Task 5: Paginated Data Browse Endpoints

**Files:**
- Create: `src/data_reader.py`
- Modify: `src/api.py`

All filtering and pagination is server-side. For the current ~5k Q&A scale, a single-pass stream over the JSONL file per request is acceptable (<50ms). If the dataset grows past 50k, add a file-watch cache (noted in comments).

**Step 1: Create `src/data_reader.py`**

```python
import json
from pathlib import Path
from typing import Any


BASE = Path(__file__).parent.parent / "data"


def _iter_jsonl(filepath: Path):
    if not filepath.exists():
        return
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _matches(item: dict, filters: dict, search: str) -> bool:
    for key, value in filters.items():
        if value and item.get(key) != value:
            return False
    if search:
        haystack = (item.get("question", "") + " " + item.get("text", "")).lower()
        if search.lower() not in haystack:
            return False
    return True


def read_jsonl_paged(
    filepath: Path,
    page: int = 1,
    size: int = 20,
    filters: dict | None = None,
    search: str = "",
) -> dict[str, Any]:
    filters = filters or {}
    size = min(size, 100)   # cap at 100 per page
    skip = (page - 1) * size
    items: list[dict] = []
    total = 0
    collected = 0

    for item in _iter_jsonl(filepath):
        if not _matches(item, filters, search):
            continue
        total += 1
        if collected < skip:
            collected += 1
            continue
        if len(items) < size:
            items.append(item)
            collected += 1

    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "pages": max(1, (total + size - 1) // size),
    }


def get_distinct_values(filepath: Path, field: str) -> list[str]:
    """Return sorted unique values for a field (for filter dropdowns)."""
    values: set[str] = set()
    for item in _iter_jsonl(filepath):
        v = item.get(field)
        if v:
            values.add(str(v))
    return sorted(values)
```

**Step 2: Add data endpoints to `src/api.py`**

```python
from src.data_reader import read_jsonl_paged, get_distinct_values, BASE

router_data = APIRouter(prefix="/api/data", tags=["data"])

@router_data.get("/chunks")
async def list_chunks(
    page: int = 1, size: int = 20,
    doc_type: str = "", insurance_type: str = "", search: str = "",
):
    filepath = BASE / "chunks" / "chunks.jsonl"
    filters = {k: v for k, v in {"doc_type": doc_type, "insurance_type": insurance_type}.items() if v}
    return read_jsonl_paged(filepath, page, size, filters, search)

@router_data.get("/qa")
async def list_qa(
    page: int = 1, size: int = 20,
    status: str = "passed",
    insurance_type: str = "", difficulty: str = "",
    question_type: str = "", generation_method: str = "",
    search: str = "",
):
    file_map = {
        "passed":   BASE / "qc_results" / "qa_passed.jsonl",
        "rejected": BASE / "qc_results" / "qa_rejected.jsonl",
        "final":    BASE / "final" / "qa_final.jsonl",
    }
    filepath = file_map.get(status, file_map["passed"])
    filters = {k: v for k, v in {
        "insurance_type": insurance_type,
        "difficulty": difficulty,
        "question_type": question_type,
        "generation_method": generation_method,
    }.items() if v}
    return read_jsonl_paged(filepath, page, size, filters, search)

@router_data.get("/qa/filters")
async def qa_filter_options():
    """Return distinct values for each filterable field."""
    filepath = BASE / "qc_results" / "qa_passed.jsonl"
    return {
        "insurance_types":      get_distinct_values(filepath, "insurance_type"),
        "difficulties":         get_distinct_values(filepath, "difficulty"),
        "question_types":       get_distinct_values(filepath, "question_type"),
        "generation_methods":   get_distinct_values(filepath, "generation_method"),
    }

@router_data.get("/reports/overview")
async def reports_overview():
    def read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    qc_report   = read_json(BASE / "qc_results" / "qc_report.json")
    pre_report  = read_json(BASE / "chunks" / "preprocess_report.json")

    seeds_count = sum(
        sum(1 for _ in open(f, encoding="utf-8"))
        for f in (BASE / "seeds").glob("seed_*.jsonl")
        if f.exists()
    )
    expanded_count = sum(
        sum(1 for _ in open(f, encoding="utf-8"))
        for f in (BASE / "expanded").glob("*.jsonl")
        if f.exists() and not f.name.endswith("_report.json")
    )

    return {
        "preprocessing": pre_report,
        "seeds_total": seeds_count,
        "expanded_total": expanded_count,
        "qc": qc_report,
    }
```

Uncomment `app.include_router(router_data)`.

**Step 3: Test**

```bash
curl "http://localhost:8000/api/data/qa?page=1&size=3&insurance_type=交强险"
# Expected: {"items":[...], "total":N, "pages":M}

curl "http://localhost:8000/api/data/qa/filters"
# Expected: {"insurance_types":[...], "difficulties":[...], ...}
```

**Step 4: Commit**

```bash
git commit -m "feat: add paginated JSONL browse endpoints for chunks and Q&A"
```

---

## Task 6: Frontend — Full Single-Page App

**Files:**
- Modify: `frontend/index.html` (complete replacement)

This is the main frontend task. The HTML file uses Vue 3 Composition API via CDN script tag. It has five "pages" rendered by a simple `currentPage` reactive ref — no router library needed.

**Pages:** Documents | Pipeline | Chunks | Q&A Browser | Reports

**Layout:**
```
┌─────────────────────────────────────────┐
│  Header: title + pipeline status dot    │
├──────────┬──────────────────────────────│
│ Sidebar  │  Main content area           │
│ • Docs   │                              │
│ • Pipe   │                              │
│ • Chunks │                              │
│ • Q&A    │                              │
│ • Report │                              │
└──────────┴──────────────────────────────┘
```

**Step 1: Write `frontend/index.html`**

Full implementation (see detailed code below). Key patterns:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Insurance QA Pipeline</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Chart.js CDN -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <!-- Vue 3 CDN -->
  <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
</head>
<body class="bg-gray-50 text-gray-800">

<div id="app">
  <!-- Header -->
  <header class="bg-white shadow-sm px-6 py-3 flex items-center justify-between">
    <h1 class="text-xl font-bold text-blue-700">Insurance QA Pipeline</h1>
    <!-- Pipeline status indicator: green pulse if idle, spinner if running -->
    <div class="flex items-center gap-2 text-sm">
      <span v-if="pipelineStatus.is_running"
            class="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></span>
      <span v-else class="w-3 h-3 rounded-full bg-green-400"></span>
      <span>{{ pipelineStatus.is_running ? '运行中…' : '就绪' }}</span>
    </div>
  </header>

  <div class="flex">
    <!-- Sidebar -->
    <nav class="w-48 min-h-screen bg-white border-r pt-4 px-3 flex flex-col gap-1">
      <button v-for="item in navItems" :key="item.key"
              @click="currentPage = item.key"
              :class="['w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                       currentPage === item.key
                         ? 'bg-blue-50 text-blue-700'
                         : 'text-gray-600 hover:bg-gray-100']">
        {{ item.icon }} {{ item.label }}
      </button>
    </nav>

    <!-- Main content -->
    <main class="flex-1 p-6 overflow-auto">
      <component :is="pages[currentPage]" />
    </main>
  </div>
</div>

<script>
const { createApp, ref, reactive, onMounted, onUnmounted, watch, computed } = Vue;

// ─── Page: Documents ─────────────────────────────────────────────────────────
const PageDocuments = {
  template: `
    <div>
      <h2 class="text-lg font-semibold mb-4">文档管理</h2>

      <!-- Drop zone -->
      <div @dragover.prevent @drop.prevent="onDrop"
           @click="$refs.fileInput.click()"
           class="border-2 border-dashed border-blue-300 rounded-xl p-10 text-center cursor-pointer hover:bg-blue-50 transition mb-6">
        <p class="text-blue-500 text-sm">点击或拖拽文件到此处上传</p>
        <p class="text-gray-400 text-xs mt-1">支持 PDF / DOCX / TXT / MD</p>
        <input ref="fileInput" type="file" multiple accept=".pdf,.docx,.doc,.txt,.md"
               class="hidden" @change="onFileSelect" />
      </div>

      <!-- Upload progress -->
      <div v-if="uploading" class="mb-4 text-sm text-blue-600 flex items-center gap-2">
        <span class="animate-spin w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full"></span>
        上传中…
      </div>

      <!-- File list -->
      <table v-if="files.length" class="w-full text-sm">
        <thead class="text-left text-gray-500 border-b">
          <tr>
            <th class="pb-2 font-medium">文件名</th>
            <th class="pb-2 font-medium">大小</th>
            <th class="pb-2 font-medium">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="f in files" :key="f.relative_path" class="border-b hover:bg-gray-50">
            <td class="py-2 pr-4">{{ f.name }}</td>
            <td class="py-2 pr-4 text-gray-500">{{ formatSize(f.size) }}</td>
            <td class="py-2">
              <button @click="deleteFile(f.relative_path)"
                      class="text-red-400 hover:text-red-600 text-xs">删除</button>
            </td>
          </tr>
        </tbody>
      </table>
      <p v-else class="text-gray-400 text-sm">暂无文件，请上传文档</p>

      <!-- Start preprocess button -->
      <button v-if="files.length"
              @click="startPreprocess"
              class="mt-6 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 transition">
        开始预处理 →
      </button>
    </div>
  `,
  setup() {
    const files = ref([]);
    const uploading = ref(false);

    async function loadFiles() {
      const r = await fetch('/api/documents/');
      const d = await r.json();
      files.value = d.files;
    }

    async function uploadFiles(fileList) {
      uploading.value = true;
      for (const file of fileList) {
        const form = new FormData();
        form.append('file', file);
        await fetch('/api/documents/upload', { method: 'POST', body: form });
      }
      uploading.value = false;
      loadFiles();
    }

    function onDrop(e) { uploadFiles(e.dataTransfer.files); }
    function onFileSelect(e) { uploadFiles(e.target.files); }

    async function deleteFile(path) {
      if (!confirm('确认删除？')) return;
      await fetch(\`/api/documents/\${path}\`, { method: 'DELETE' });
      loadFiles();
    }

    async function startPreprocess() {
      await fetch('/api/pipeline/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step: 'preprocess' })
      });
      alert('预处理已启动，请前往「流水线」页面查看进度');
    }

    function formatSize(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / 1024 / 1024).toFixed(1) + ' MB';
    }

    onMounted(loadFiles);
    return { files, uploading, onDrop, onFileSelect, deleteFile, startPreprocess, formatSize };
  }
};

// ─── Page: Pipeline ──────────────────────────────────────────────────────────
const PagePipeline = {
  template: `
    <div>
      <h2 class="text-lg font-semibold mb-6">流水线控制</h2>

      <!-- Step cards -->
      <div class="flex gap-4 mb-8 flex-wrap">
        <div v-for="(step, key) in status.steps" :key="key"
             class="flex-1 min-w-40 bg-white border rounded-xl p-4 shadow-sm">
          <div class="flex items-center gap-2 mb-2">
            <span :class="statusDot(step.status)"></span>
            <span class="font-medium text-sm">{{ step.label }}</span>
          </div>
          <div class="text-xs text-gray-500 mb-1">{{ statusText(step.status) }}</div>
          <div v-if="step.status === 'running'"
               class="w-full bg-gray-100 rounded h-1.5 mt-2">
            <div class="bg-blue-500 h-1.5 rounded transition-all duration-500 animate-pulse"
                 style="width: 60%"></div>
          </div>
          <div v-if="step.output_count" class="text-xs text-green-600 mt-1">
            输出: {{ step.output_count.toLocaleString() }} 条
          </div>
          <div v-if="step.error" class="text-xs text-red-500 mt-1 truncate" :title="step.error">
            {{ step.error }}
          </div>
        </div>
      </div>

      <!-- Control buttons -->
      <div class="flex gap-3 mb-6 flex-wrap">
        <button @click="start('all')" :disabled="status.is_running"
                class="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm disabled:opacity-50 hover:bg-blue-700 transition">
          运行全流程
        </button>
        <button v-for="s in ['preprocess','seed','expand','qc']" :key="s"
                @click="start(s)" :disabled="status.is_running"
                class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm disabled:opacity-50 hover:bg-gray-50 transition">
          仅运行: {{ stepLabel(s) }}
        </button>
      </div>

      <!-- Log output -->
      <div class="bg-gray-900 rounded-xl p-4 font-mono text-xs text-green-300 h-56 overflow-y-auto"
           ref="logBox">
        <div v-if="!logs.length" class="text-gray-500">等待日志…</div>
        <div v-for="(line, i) in logs" :key="i"
             :class="line.level === 'ERROR' ? 'text-red-400' : line.level === 'WARNING' ? 'text-yellow-300' : ''">
          {{ line.message }}
        </div>
      </div>
    </div>
  `,
  props: ['status'],
  setup(props) {
    const logs = ref([]);
    const logBox = ref(null);
    let es = null;

    function connectSSE() {
      es = new EventSource('/api/pipeline/events');
      es.onmessage = (e) => {
        const event = JSON.parse(e.data);
        if (event.type === 'log') {
          logs.value.push(event);
          if (logs.value.length > 200) logs.value.shift();
          Vue.nextTick(() => {
            if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight;
          });
        }
      };
    }

    async function start(step) {
      logs.value = [];
      await fetch('/api/pipeline/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step })
      });
    }

    function statusDot(s) {
      const map = { idle: 'w-2 h-2 rounded-full bg-gray-300', running: 'w-2 h-2 rounded-full bg-blue-500 animate-pulse', done: 'w-2 h-2 rounded-full bg-green-500', error: 'w-2 h-2 rounded-full bg-red-500' };
      return map[s] || map.idle;
    }
    function statusText(s) {
      return { idle: '待运行', running: '运行中…', done: '已完成', error: '出错' }[s] || s;
    }
    function stepLabel(s) {
      return { preprocess: '预处理', seed: '种子生成', expand: '批量扩展', qc: '质检' }[s] || s;
    }

    onMounted(connectSSE);
    onUnmounted(() => { if (es) es.close(); });
    return { logs, logBox, start, statusDot, statusText, stepLabel };
  }
};

// ─── Page: Q&A Browser ───────────────────────────────────────────────────────
const PageQA = {
  template: `
    <div>
      <h2 class="text-lg font-semibold mb-4">Q&A 浏览器</h2>

      <!-- Filters -->
      <div class="bg-white border rounded-xl p-4 mb-4 flex flex-wrap gap-3 items-end">
        <div v-for="(opts, key) in filterOpts" :key="key" class="flex flex-col gap-1">
          <label class="text-xs text-gray-500">{{ filterLabel(key) }}</label>
          <select v-model="filters[key]" @change="goPage(1)"
                  class="border rounded px-2 py-1.5 text-sm min-w-28">
            <option value="">全部</option>
            <option v-for="o in opts" :key="o" :value="o">{{ o }}</option>
          </select>
        </div>
        <!-- Status -->
        <div class="flex flex-col gap-1">
          <label class="text-xs text-gray-500">状态</label>
          <select v-model="filters.status" @change="goPage(1)"
                  class="border rounded px-2 py-1.5 text-sm">
            <option value="passed">通过</option>
            <option value="rejected">拒绝</option>
          </select>
        </div>
        <!-- Search -->
        <div class="flex flex-col gap-1 flex-1 min-w-48">
          <label class="text-xs text-gray-500">搜索问题</label>
          <input v-model="searchText" @keyup.enter="goPage(1)" type="text"
                 placeholder="输入关键词…"
                 class="border rounded px-3 py-1.5 text-sm" />
        </div>
        <!-- Page size -->
        <div class="flex flex-col gap-1">
          <label class="text-xs text-gray-500">每页</label>
          <select v-model.number="pageSize" @change="goPage(1)"
                  class="border rounded px-2 py-1.5 text-sm">
            <option :value="20">20</option>
            <option :value="50">50</option>
            <option :value="100">100</option>
          </select>
        </div>
        <button @click="resetFilters" class="text-xs text-gray-400 hover:text-gray-600 underline self-end pb-2">重置</button>
      </div>

      <!-- Results count -->
      <div class="text-sm text-gray-500 mb-2">
        共 {{ result.total.toLocaleString() }} 条，第 {{ result.page }} / {{ result.pages }} 页
      </div>

      <!-- Table -->
      <div class="bg-white border rounded-xl overflow-hidden mb-4">
        <table class="w-full text-sm">
          <thead class="bg-gray-50 text-gray-500 text-xs border-b">
            <tr>
              <th class="text-left px-4 py-2 font-medium w-1/2">问题</th>
              <th class="text-left px-4 py-2 font-medium">险种</th>
              <th class="text-left px-4 py-2 font-medium">难度</th>
              <th class="text-left px-4 py-2 font-medium">类型</th>
              <th class="text-left px-4 py-2 font-medium">来源</th>
            </tr>
          </thead>
          <tbody>
            <template v-for="(item, i) in result.items" :key="item.id || i">
              <tr @click="toggleExpand(i)"
                  class="border-b cursor-pointer hover:bg-blue-50 transition">
                <td class="px-4 py-2 text-gray-800">{{ truncate(item.question, 80) }}</td>
                <td class="px-4 py-2 text-gray-500 text-xs">{{ item.insurance_type || '—' }}</td>
                <td class="px-4 py-2">
                  <span :class="diffBadge(item.difficulty)">{{ item.difficulty || '—' }}</span>
                </td>
                <td class="px-4 py-2 text-gray-500 text-xs">{{ item.question_type || '—' }}</td>
                <td class="px-4 py-2 text-gray-400 text-xs">{{ item.generation_method || '—' }}</td>
              </tr>
              <tr v-if="expanded === i">
                <td colspan="5" class="px-6 py-4 bg-blue-50 text-sm">
                  <div class="font-medium mb-1 text-blue-800">{{ item.question }}</div>
                  <div class="text-gray-700 whitespace-pre-wrap mb-3">{{ item.answer }}</div>
                  <div v-if="item.tags?.length" class="flex flex-wrap gap-1">
                    <span v-for="t in item.tags" :key="t"
                          class="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">{{ t }}</span>
                  </div>
                  <div class="mt-2 text-xs text-gray-400">
                    来源: {{ item.source_doc || '—' }}
                    | 批次: {{ item.batch_id || '—' }}
                  </div>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
        <div v-if="loading" class="text-center py-6 text-gray-400 text-sm">加载中…</div>
        <div v-if="!loading && !result.items.length" class="text-center py-8 text-gray-400 text-sm">无数据</div>
      </div>

      <!-- Pagination -->
      <div class="flex items-center gap-2 text-sm">
        <button @click="goPage(1)" :disabled="result.page <= 1"
                class="px-2 py-1 border rounded disabled:opacity-30">«</button>
        <button @click="goPage(result.page - 1)" :disabled="result.page <= 1"
                class="px-3 py-1 border rounded disabled:opacity-30">‹ 上一页</button>
        <span class="px-2 text-gray-500">
          第 <input type="number" v-model.number="jumpPage" @keyup.enter="goPage(jumpPage)"
                    class="w-14 border rounded px-1 text-center" :min="1" :max="result.pages" />
          / {{ result.pages }} 页
        </span>
        <button @click="goPage(result.page + 1)" :disabled="result.page >= result.pages"
                class="px-3 py-1 border rounded disabled:opacity-30">下一页 ›</button>
        <button @click="goPage(result.pages)" :disabled="result.page >= result.pages"
                class="px-2 py-1 border rounded disabled:opacity-30">»</button>
      </div>
    </div>
  `,
  setup() {
    const filters = reactive({
      status: 'passed', insurance_type: '', difficulty: '',
      question_type: '', generation_method: '',
    });
    const searchText = ref('');
    const pageSize = ref(20);
    const jumpPage = ref(1);
    const expanded = ref(null);
    const loading = ref(false);
    const filterOpts = ref({ insurance_type: [], difficulty: [], question_type: [], generation_method: [] });
    const result = reactive({ items: [], total: 0, page: 1, pages: 1, size: 20 });

    async function loadFilterOptions() {
      const r = await fetch('/api/data/qa/filters');
      const d = await r.json();
      filterOpts.value = {
        insurance_type: d.insurance_types,
        difficulty: d.difficulties,
        question_type: d.question_types,
        generation_method: d.generation_methods,
      };
    }

    async function loadPage(page = 1) {
      loading.value = true;
      expanded.value = null;
      const params = new URLSearchParams({
        page, size: pageSize.value, status: filters.status,
        insurance_type: filters.insurance_type,
        difficulty: filters.difficulty,
        question_type: filters.question_type,
        generation_method: filters.generation_method,
        search: searchText.value,
      });
      const r = await fetch(\`/api/data/qa?\${params}\`);
      const d = await r.json();
      Object.assign(result, d);
      jumpPage.value = page;
      loading.value = false;
    }

    function goPage(p) {
      const n = Math.max(1, Math.min(p, result.pages));
      loadPage(n);
    }

    function toggleExpand(i) { expanded.value = expanded.value === i ? null : i; }
    function truncate(s, n) { return s && s.length > n ? s.slice(0, n) + '…' : s; }
    function diffBadge(d) {
      return { '入门': 'px-1.5 py-0.5 rounded text-xs bg-green-100 text-green-700', '进阶': 'px-1.5 py-0.5 rounded text-xs bg-yellow-100 text-yellow-700', '专业': 'px-1.5 py-0.5 rounded text-xs bg-red-100 text-red-700' }[d] || 'text-xs text-gray-500';
    }
    function filterLabel(k) {
      return { insurance_type: '险种', difficulty: '难度', question_type: '问题类型', generation_method: '生成方式' }[k] || k;
    }
    function resetFilters() {
      Object.assign(filters, { status: 'passed', insurance_type: '', difficulty: '', question_type: '', generation_method: '' });
      searchText.value = '';
      loadPage(1);
    }

    onMounted(() => { loadFilterOptions(); loadPage(1); });
    return { filters, searchText, pageSize, jumpPage, expanded, loading, filterOpts, result, goPage, toggleExpand, truncate, diffBadge, filterLabel, resetFilters };
  }
};

// ─── Page: Chunks ─────────────────────────────────────────────────────────
const PageChunks = {
  template: `
    <div>
      <h2 class="text-lg font-semibold mb-4">文档 Chunks</h2>
      <div class="flex gap-3 mb-4 flex-wrap">
        <select v-model="filters.doc_type" @change="load(1)" class="border rounded px-2 py-1.5 text-sm">
          <option value="">文档类型: 全部</option>
          <option v-for="t in docTypes" :key="t" :value="t">{{ t }}</option>
        </select>
        <select v-model="filters.insurance_type" @change="load(1)" class="border rounded px-2 py-1.5 text-sm">
          <option value="">险种: 全部</option>
          <option v-for="t in insTypes" :key="t" :value="t">{{ t }}</option>
        </select>
        <input v-model="search" @keyup.enter="load(1)" placeholder="搜索内容…"
               class="border rounded px-3 py-1.5 text-sm flex-1 min-w-48" />
      </div>
      <div class="text-sm text-gray-500 mb-2">共 {{ result.total }} 条</div>
      <div class="space-y-2">
        <div v-for="(c, i) in result.items" :key="c.chunk_id || i"
             @click="expanded = expanded === i ? null : i"
             class="bg-white border rounded-lg p-3 cursor-pointer hover:border-blue-300 transition">
          <div class="flex justify-between items-start">
            <div>
              <span class="text-xs font-mono text-gray-400 mr-2">{{ c.chunk_id }}</span>
              <span class="text-xs bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded">{{ c.doc_type }}</span>
              <span v-if="c.section_title" class="text-xs text-gray-500 ml-2">{{ c.section_title }}</span>
            </div>
            <span class="text-xs text-gray-400">{{ c.char_count }} 字</span>
          </div>
          <p class="text-sm text-gray-600 mt-1 line-clamp-2">{{ c.text }}</p>
          <div v-if="expanded === i" class="mt-2 text-sm text-gray-700 whitespace-pre-wrap border-t pt-2">
            {{ c.text }}
          </div>
        </div>
      </div>
      <!-- Pagination -->
      <div class="flex items-center gap-2 text-sm mt-4">
        <button @click="load(result.page-1)" :disabled="result.page<=1" class="px-3 py-1 border rounded disabled:opacity-30">‹</button>
        <span class="text-gray-500">第 {{ result.page }} / {{ result.pages }} 页</span>
        <button @click="load(result.page+1)" :disabled="result.page>=result.pages" class="px-3 py-1 border rounded disabled:opacity-30">›</button>
      </div>
    </div>
  `,
  setup() {
    const filters = reactive({ doc_type: '', insurance_type: '' });
    const search = ref('');
    const expanded = ref(null);
    const result = reactive({ items: [], total: 0, page: 1, pages: 1 });
    const docTypes = ref([]);
    const insTypes = ref([]);

    async function load(page = 1) {
      expanded.value = null;
      const params = new URLSearchParams({ page, size: 20, doc_type: filters.doc_type, insurance_type: filters.insurance_type, search: search.value });
      const r = await fetch(\`/api/data/chunks?\${params}\`);
      Object.assign(result, await r.json());
    }

    onMounted(async () => {
      await load(1);
      // Extract unique filter values from loaded items
      const all = result.items;
      docTypes.value = [...new Set(all.map(c => c.doc_type).filter(Boolean))];
      insTypes.value = [...new Set(all.map(c => c.insurance_type).filter(Boolean))];
    });

    return { filters, search, expanded, result, docTypes, insTypes, load };
  }
};

// ─── Page: Reports ────────────────────────────────────────────────────────
const PageReports = {
  template: `
    <div>
      <h2 class="text-lg font-semibold mb-4">数据报告</h2>

      <!-- Summary cards -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div class="bg-white border rounded-xl p-4 text-center shadow-sm">
          <div class="text-2xl font-bold text-blue-600">{{ (overview.preprocessing?.total_chunks || 0).toLocaleString() }}</div>
          <div class="text-xs text-gray-500 mt-1">文档 Chunks</div>
        </div>
        <div class="bg-white border rounded-xl p-4 text-center shadow-sm">
          <div class="text-2xl font-bold text-purple-600">{{ (overview.seeds_total || 0).toLocaleString() }}</div>
          <div class="text-xs text-gray-500 mt-1">种子 Q&A</div>
        </div>
        <div class="bg-white border rounded-xl p-4 text-center shadow-sm">
          <div class="text-2xl font-bold text-yellow-600">{{ (overview.expanded_total || 0).toLocaleString() }}</div>
          <div class="text-xs text-gray-500 mt-1">扩展 Q&A</div>
        </div>
        <div class="bg-white border rounded-xl p-4 text-center shadow-sm">
          <div class="text-2xl font-bold text-green-600">{{ (overview.qc?.passed_total || 0).toLocaleString() }}</div>
          <div class="text-xs text-gray-500 mt-1">质检通过</div>
        </div>
      </div>

      <!-- Charts row -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-white border rounded-xl p-4 shadow-sm">
          <h3 class="text-sm font-medium mb-3">险种分布</h3>
          <canvas ref="insChart" height="220"></canvas>
        </div>
        <div class="bg-white border rounded-xl p-4 shadow-sm">
          <h3 class="text-sm font-medium mb-3">难度分布</h3>
          <canvas ref="diffChart" height="220"></canvas>
        </div>
      </div>

      <!-- QC rejection reasons -->
      <div v-if="qcReasons.length" class="mt-6 bg-white border rounded-xl p-4 shadow-sm">
        <h3 class="text-sm font-medium mb-3">质检拒绝原因</h3>
        <table class="w-full text-sm">
          <tr v-for="r in qcReasons" :key="r.reason" class="border-b">
            <td class="py-1.5 text-gray-700">{{ r.reason }}</td>
            <td class="py-1.5 text-right text-gray-500">{{ r.count }}</td>
          </tr>
        </table>
      </div>
    </div>
  `,
  setup() {
    const overview = ref({});
    const qcReasons = ref([]);
    const insChart = ref(null);
    const diffChart = ref(null);

    async function loadData() {
      const r = await fetch('/api/data/reports/overview');
      overview.value = await r.json();

      // Build rejection reasons from qc report
      const reject = overview.value.qc?.rejection_reasons || {};
      qcReasons.value = Object.entries(reject)
        .map(([reason, count]) => ({ reason, count }))
        .sort((a, b) => b.count - a.count);

      Vue.nextTick(() => renderCharts());
    }

    function renderCharts() {
      const insDist = overview.value.qc?.insurance_type_distribution || {};
      const diffDist = overview.value.qc?.difficulty_distribution || {};

      if (insChart.value && Object.keys(insDist).length) {
        const entries = Object.entries(insDist).sort((a, b) => b[1] - a[1]).slice(0, 10);
        new Chart(insChart.value, {
          type: 'bar',
          data: { labels: entries.map(e => e[0]), datasets: [{ data: entries.map(e => e[1]), backgroundColor: '#3b82f6' }] },
          options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } } } }
        });
      }

      if (diffChart.value && Object.keys(diffDist).length) {
        new Chart(diffChart.value, {
          type: 'doughnut',
          data: {
            labels: Object.keys(diffDist),
            datasets: [{ data: Object.values(diffDist), backgroundColor: ['#22c55e', '#eab308', '#ef4444'] }]
          },
          options: { plugins: { legend: { position: 'bottom' } } }
        });
      }
    }

    onMounted(loadData);
    return { overview, qcReasons, insChart, diffChart };
  }
};

// ─── Root App ─────────────────────────────────────────────────────────────
createApp({
  components: { PageDocuments, PagePipeline, PageChunks, PageQA, PageReports },
  setup() {
    const currentPage = ref('documents');
    const pipelineStatus = reactive({ is_running: false, steps: {} });

    const navItems = [
      { key: 'documents', label: '文档上传', icon: '📁' },
      { key: 'pipeline',  label: '流水线',   icon: '⚙️' },
      { key: 'chunks',    label: 'Chunks',   icon: '📦' },
      { key: 'qa',        label: 'Q&A 浏览', icon: '💬' },
      { key: 'reports',   label: '数据报告', icon: '📊' },
    ];

    const pages = {
      documents: PageDocuments,
      pipeline:  PagePipeline,
      chunks:    PageChunks,
      qa:        PageQA,
      reports:   PageReports,
    };

    // Global SSE for header status indicator
    let es = null;
    function connectGlobalSSE() {
      es = new EventSource('/api/pipeline/events');
      es.onmessage = (e) => {
        const event = JSON.parse(e.data);
        if (event.type === 'init') Object.assign(pipelineStatus, event.state);
        else if (event.type === 'step_update') pipelineStatus.is_running = true;
        // Refresh status periodically
      };
      es.onerror = () => setTimeout(connectGlobalSSE, 3000);
    }

    // Poll status every 3s as a reliable fallback
    let pollTimer = null;
    async function pollStatus() {
      const r = await fetch('/api/pipeline/status');
      Object.assign(pipelineStatus, await r.json());
    }

    onMounted(() => {
      connectGlobalSSE();
      pollStatus();
      pollTimer = setInterval(pollStatus, 3000);
    });
    onUnmounted(() => {
      if (es) es.close();
      clearInterval(pollTimer);
    });

    return { currentPage, navItems, pages, pipelineStatus };
  },
  template: `
    <div>
      <header class="bg-white shadow-sm px-6 py-3 flex items-center justify-between sticky top-0 z-10">
        <h1 class="text-xl font-bold text-blue-700">Insurance QA Pipeline</h1>
        <div class="flex items-center gap-2 text-sm">
          <span v-if="pipelineStatus.is_running"
                class="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></span>
          <span v-else class="w-3 h-3 rounded-full bg-green-400"></span>
          <span class="text-gray-600">{{ pipelineStatus.is_running ? '运行中…' : '就绪' }}</span>
        </div>
      </header>
      <div class="flex min-h-screen">
        <nav class="w-48 bg-white border-r pt-4 px-3 flex flex-col gap-1 flex-shrink-0">
          <button v-for="item in navItems" :key="item.key"
                  @click="currentPage = item.key"
                  :class="['w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                           currentPage === item.key ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-100']">
            {{ item.icon }} {{ item.label }}
          </button>
        </nav>
        <main class="flex-1 p-6 overflow-auto">
          <component :is="pages[currentPage]" :status="pipelineStatus" />
        </main>
      </div>
    </div>
  `
}).mount('#app');
</script>
</body>
</html>
```

**Step 2: Verify in browser**

```bash
uvicorn src.api:app --reload --port 8000
open http://localhost:8000
```

Walk through each page:
- Documents: upload a `.md` file, verify it appears in the list
- Pipeline: click "运行全流程", verify SSE log messages appear
- Q&A: verify pagination controls work, filters narrow results
- Reports: verify summary cards show numbers, charts render

**Step 3: Commit**

```bash
git add frontend/index.html
git commit -m "feat: complete frontend SPA with all 5 pages"
```

---

## Task 7: Wire Up All Routers + Startup Script

**Files:**
- Modify: `src/api.py` (uncomment all `include_router` lines)
- Create: `start.sh`

**Step 1: Final `src/api.py` with all routers included and CORS for dev**

```python
import sys, json, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, APIRouter, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio

from src.pipeline_state import pipeline_state
from src.pipeline_runner import run_step
from src.data_reader import read_jsonl_paged, get_distinct_values, BASE

app = FastAPI(title="Insurance QA Pipeline")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- Documents router ----------
# ... (as defined in Task 3)

# ---------- Pipeline router ----------
# ... (as defined in Task 4)

# ---------- Data router ----------
# ... (as defined in Task 5)

app.include_router(router_docs)
app.include_router(router_pipeline)
app.include_router(router_data)

frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")
```

**Step 2: Create `start.sh`**

```bash
#!/bin/bash
cd "$(dirname "$0")"
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

```bash
chmod +x start.sh
```

**Step 3: End-to-end test**

1. Open http://localhost:8000
2. Go to Documents → upload a new `.md` file → confirm it appears
3. Go to Pipeline → click "仅运行: 预处理" → confirm log messages stream in
4. Go to Q&A → change page size to 50 → click next page → confirm pagination
5. Go to Reports → confirm charts render

**Step 4: Final commit**

```bash
git add src/api.py start.sh
git commit -m "feat: wire all API routers and add start script"
```

---

## Verification Checklist

- [ ] `uvicorn src.api:app --reload --port 8000` starts without errors
- [ ] `http://localhost:8000` loads the Vue SPA
- [ ] File upload: drag `.md` → appears in list
- [ ] Pipeline start: SSE log messages stream correctly
- [ ] Q&A Browser: filters + search narrow results server-side
- [ ] Q&A Browser: page size 20/50/100 works; jump-to-page works
- [ ] Row expand: full Q&A content visible
- [ ] Reports: summary cards show real numbers from `qc_report.json`
- [ ] Header status dot turns to spinner when pipeline is running

---

## Running the App

```bash
pip install fastapi "uvicorn[standard]" python-multipart sse-starlette
./start.sh
# or
uvicorn src.api:app --reload --port 8000
```
