import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


STATE_FILE = Path(__file__).parent.parent / "data" / "pipeline_state.json"


class StepStatus(str, Enum):
    IDLE    = "idle"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"
    STOPPED = "stopped"


@dataclass
class StepState:
    key:          str
    label:        str
    status:       StepStatus = StepStatus.IDLE
    progress:     int = 0
    message:      str = ""
    output_count: int = 0
    error:        str = ""


class PipelineState:
    def __init__(self):
        self._lock        = threading.Lock()
        self.steps: dict[str, StepState] = {
            "preprocess": StepState("preprocess", "文档预处理"),
            "seed":       StepState("seed",       "种子生成"),
            "seed_dedup": StepState("seed_dedup", "种子去重"),
            "expand":     StepState("expand",     "批量扩展"),
            "qc":         StepState("qc",         "质量检查"),
        }
        self.current_step: Optional[str] = None
        self.is_running: bool = False
        self._queue: Optional[asyncio.Queue] = None
        self._cancel_event = threading.Event()
        self._load()

    def _save(self):
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self):
        try:
            if not STATE_FILE.exists():
                return
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                # 进程崩溃重启后，is_running 和 current_step 始终重置
                self.is_running = False
                self.current_step = None
                for key, step_data in data.get("steps", {}).items():
                    if key in self.steps:
                        s = self.steps[key]
                        status = step_data.get("status", "idle")
                        # running 状态重置为 idle（崩溃恢复）
                        if status == "running":
                            status = "idle"
                        s.status = StepStatus(status)
                        s.progress = step_data.get("progress", 0)
                        s.output_count = step_data.get("output_count", 0)
                        s.message = step_data.get("message", "")
                        s.error = step_data.get("error", "")
        except Exception:
            pass

    def request_stop(self):
        """Signal the running pipeline thread to stop after current step."""
        self._cancel_event.set()
        self.log("⏹️ 收到停止请求，当前步骤完成后将停止", "WARNING")

    def is_stop_requested(self) -> bool:
        return self._cancel_event.is_set()

    def clear_stop(self):
        self._cancel_event.clear()

    def _get_queue(self) -> Optional[asyncio.Queue]:
        if self._queue is None:
            try:
                asyncio.get_running_loop()
                self._queue = asyncio.Queue()
            except RuntimeError:
                pass
        return self._queue

    def reset_queue(self):
        """Flush stale events; call before a new SSE consumer connects."""
        self._queue = asyncio.Queue()

    def update_step(self, step: str, **kwargs):
        with self._lock:
            s = self.steps[step]
            for k, v in kwargs.items():
                setattr(s, k, v)
        self._save()
        self._emit({"type": "step_update", "step": step, **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, bool, float))}})

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
                "is_running":   self.is_running,
                "steps": {
                    k: {
                        "key":          v.key,
                        "label":        v.label,
                        "status":       v.status,
                        "progress":     v.progress,
                        "message":      v.message,
                        "output_count": v.output_count,
                        "error":        v.error,
                    }
                    for k, v in self.steps.items()
                },
            }


pipeline_state = PipelineState()


class PipelineLogHandler(logging.Handler):
    """Attached to root logger during pipeline runs; forwards all log records to the SSE queue."""

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        pipeline_state.log(msg, record.levelname)
