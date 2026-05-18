"""
llm_logger.py
LLM 调用日志：记录每次调用的输入 prompt、输出 response、token 用量、耗时和成本。

日志文件：data/llm_logs/llm_calls.jsonl（每行一条 JSON 记录）
"""

import json
import uuid
import threading
import asyncio
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_DIR  = Path(__file__).parent.parent / "data" / "llm_logs"
_LOG_FILE = _LOG_DIR / "llm_calls.jsonl"
_lock     = threading.Lock()

# 每行记录的字段：
# call_id, timestamp, stage, provider, model
# prompt (full), system (full), response (full)
# input_tokens, output_tokens, cost_yuan, duration_ms
# status ("success" | "failed"), error


def _ensure_dir():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(record: dict):
    """线程安全地追加一条日志记录（同步调用路径）。"""
    try:
        _ensure_dir()
        with _lock:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"写入 LLM 日志失败: {e}")


async def awrite_log(record: dict):
    """异步路径下的日志写入（运行在线程池中，不阻塞事件循环）。"""
    await asyncio.to_thread(write_log, record)


def make_record(
    stage: str,
    provider: str,
    model: str,
    prompt: str,
    system: str,
    response: str,
    input_tokens: int,
    output_tokens: int,
    cost_yuan: float,
    duration_ms: int,
    status: str = "success",
    error: str = "",
) -> dict:
    return {
        "call_id":      str(uuid.uuid4())[:8],
        "timestamp":    datetime.now().isoformat(timespec="milliseconds"),
        "stage":        stage,
        "provider":     provider,
        "model":        model,
        "prompt":       prompt,
        "system":       system,
        "response":     response,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "cost_yuan":    round(cost_yuan, 6),
        "duration_ms":  duration_ms,
        "status":       status,
        "error":        error,
    }
