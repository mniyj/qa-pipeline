import logging
import threading

# Use src. prefix to share the same module instance as api.py.
# Bare imports (from pipeline_state) would create a second instance.
from src.pipeline_state import pipeline_state, PipelineLogHandler
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
    Execute one or all pipeline steps in a background thread.
    Injects PipelineLogHandler so every logger.info() becomes an SSE event.
    """
    if not _run_lock.acquire(blocking=False):
        return  # already running — silently skip

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
                    report = cmd_preprocess(config) or {}
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("total_chunks", 0),
                    )

                elif s == "seed":
                    report = cmd_seed(config) or {}
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("total_generated", 0),
                    )

                elif s == "expand":
                    cmd_expand(config)
                    pipeline_state.update_step(s, status="done", progress=100)

                elif s == "qc":
                    report = cmd_qc(config) or {}
                    pipeline_state.update_step(
                        s, status="done", progress=100,
                        output_count=report.get("passed_total", 0),
                    )

            except Exception as e:
                pipeline_state.update_step(s, status="error", error=str(e))
                raise

    finally:
        root_logger.removeHandler(handler)
        pipeline_state.is_running = False
        pipeline_state.current_step = None
        _run_lock.release()
