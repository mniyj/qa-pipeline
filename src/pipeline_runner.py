import json
import logging
import threading
from pathlib import Path

# Use src. prefix to share the same module instance as api.py.
# Bare imports (from pipeline_state) would create a second instance.
from src.pipeline_state import pipeline_state, PipelineLogHandler
from orchestrator import (
    load_config,
    cmd_preprocess,
    cmd_seed,
    cmd_seed_dedup,
    cmd_expand,
    cmd_tool_routing,
    cmd_comparison,
    cmd_p2_quality,
    cmd_qc,
)

_run_lock = threading.Lock()


def _count_lines(filepath) -> int:
    p = Path(filepath)
    if not p.exists():
        return 0
    return sum(1 for _ in open(p, encoding="utf-8") if _.strip())


def _upstream_ready(config: dict, step: str) -> bool:
    """检查上游步骤是否已产生数据"""
    paths = config["storage"]["paths"]
    if step == "seed":
        return (Path(paths["chunks"]) / "chunks.jsonl").exists()
    if step == "expand":
        seeds_dir = Path(paths["seeds"])
        return seeds_dir.exists() and any(seeds_dir.glob("seed_*.jsonl"))
    if step == "qc":
        expanded_dir = Path(paths["expanded"])
        return expanded_dir.exists() and any(
            expanded_dir.glob("*.jsonl")
            and not f.name.endswith("_report.json")
            for f in expanded_dir.iterdir()
        )
    return True


def _already_done(config: dict, step: str) -> str | None:
    """
    检查步骤是否已完成（已有输出文件）。
    返回 None 表示需要执行，返回 "skip" 表示跳过。
    """
    paths = config["storage"]["paths"]
    if step == "preprocess":
        return None  # preprocess 始终检查新文件
    if step == "seed":
        seeded_file = Path(paths["seeds"]) / "seeded_chunks.json"
        if not seeded_file.exists():
            return None
        with open(seeded_file, "r") as f:
            seeded_ids = json.load(f)
        # 检查 chunks 总数 vs 已处理数
        chunks_file = Path(paths["chunks"]) / "chunks.jsonl"
        if chunks_file.exists():
            total_chunks = _count_lines(chunks_file)
            if total_chunks > 0 and len(seeded_ids) >= total_chunks:
                return "skip"
    if step == "expand":
        expanded_dir = Path(paths["expanded"])
        # 如果有已经扩展开的文件 + checkpoint 存在，表示已完成
        if expanded_dir.exists():
            exp_files = [f for f in expanded_dir.glob("*.jsonl")
                         if not f.name.endswith("_report.json")]
            if exp_files:
                return None  # 不能简单跳过，可能有新的种子文件需要每次检查
    return None


def _make_progress_cb(step: str):
    """Returns a callback(current, total) that emits progress SSE events for the given step."""
    def cb(current: int, total: int):
        if total > 0:
            pct = min(99, int(current / total * 100))
            pipeline_state.update_step(step, progress=pct)
    return cb


def run_step(step: str):
    """
    Execute one or all pipeline steps in a background thread.
    Injects PipelineLogHandler so every logger.info() becomes an SSE event.
    """
    if not _run_lock.acquire(blocking=False):
        return  # already running — silently skip

    handler = PipelineLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    steps = ["preprocess", "seed", "seed_dedup", "expand", "qc"] if step == "all" else [step]

    try:
        pipeline_state.is_running = True
        pipeline_state.clear_stop()
        pipeline_state._save()
        config = load_config()

        for s in steps:
            # 检查是否收到停止信号
            if pipeline_state.is_stop_requested():
                pipeline_state.log(f"⏹️ 已停止，跳过后续步骤: {s}", "WARNING")
                break

            # 上游依赖检查
            if not _upstream_ready(config, s):
                pipeline_state.update_step(
                    s, status="error",
                    error="上游数据未就绪，请先完成前置步骤",
                )
                continue

            pipeline_state.current_step = s
            pipeline_state.update_step(s, status="running", progress=0, error="")

            try:
                if s == "preprocess":
                    report = cmd_preprocess(config, progress_callback=_make_progress_cb(s)) or {}
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("total_chunks", 0),
                    )

                elif s == "seed":
                    report = cmd_seed(config, progress_callback=_make_progress_cb(s)) or {}
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("total_generated", 0),
                    )

                elif s == "seed_dedup":
                    report = cmd_seed_dedup(config, progress_callback=_make_progress_cb(s)) or {}
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("unique", 0),
                        message=f"过滤 {report.get('duplicates', 0)} 条重复种子",
                    )

                elif s == "expand":
                    cmd_expand(config, progress_callback=_make_progress_cb(s))
                    expanded_dir = Path(config["storage"]["paths"]["expanded"])
                    exp_count = sum(
                        _count_lines(f) for f in expanded_dir.glob("*.jsonl")
                        if not f.name.endswith("_report.json") and not f.name.startswith(".")
                    ) if expanded_dir.exists() else 0
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=exp_count,
                    )

                elif s == "tool_routing":
                    report = cmd_tool_routing(config, progress_callback=_make_progress_cb(s)) or {}
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("total", 0),
                        message=f"工具路由 {report.get('tool_routing_count', 0)} 条 + 保全变更 {report.get('policy_service_count', 0)} 条",
                    )

                elif s == "comparison":
                    report = cmd_comparison(config, progress_callback=_make_progress_cb(s)) or {}
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("total", 0),
                        message=f"对比 {report.get('comparison_count', 0)} + 误解纠正 {report.get('misconception_count', 0)} + 澄清 {report.get('clarification_count', 0)}",
                    )

                elif s == "p2_quality":
                    report = cmd_p2_quality(config, progress_callback=_make_progress_cb(s)) or {}
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("tool_routing_rephrase", 0),
                        message=f"改述 {report.get('tool_routing_rephrase', 0)} + 回溯存疑 {report.get('anchor_verify_flagged', 0)}",
                    )

                elif s == "qc":
                    report = cmd_qc(config, progress_callback=_make_progress_cb(s)) or {}
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("passed_total", 0),
                    )

                # 当前步骤完成后再次检查停止信号（"all"模式时用于跳过后续步骤）
                if pipeline_state.is_stop_requested():
                    pipeline_state.log(f"⏹️ 收到停止请求，{s} 完成后停止", "WARNING")
                    break

            except Exception as e:
                pipeline_state.update_step(s, status="error", error=str(e))
                raise

    finally:
        # 如果有停止请求但步骤仍为 running，标记为 stopped
        if pipeline_state.is_stop_requested():
            for s in steps:
                step_state = pipeline_state.steps.get(s)
                if step_state and step_state.status == "running":
                    pipeline_state.update_step(s, status="stopped", progress=0)
        root_logger.removeHandler(handler)
        pipeline_state.is_running = False
        pipeline_state.current_step = None
        pipeline_state.clear_stop()
        pipeline_state._save()
        pipeline_state._emit({"type": "pipeline_done", "state": pipeline_state.to_dict()})
        _run_lock.release()
