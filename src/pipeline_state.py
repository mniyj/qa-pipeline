import asyncio
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StepStatus(str, Enum):
    IDLE    = "idle"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"


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
            "expand":     StepState("expand",     "批量扩展"),
            "qc":         StepState("qc",         "质量检查"),
        }
        self.current_step: Optional[str] = None
        self.is_running: bool = False
        self._queue: Optional[asyncio.Queue] = None

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
