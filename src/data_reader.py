import json
from pathlib import Path
from typing import Any


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
    size    = min(max(size, 1), 100)
    skip    = (page - 1) * size
    items: list[dict] = []
    total     = 0
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
        "page":  page,
        "size":  size,
        "pages": max(1, (total + size - 1) // size),
    }


def get_distinct_values(filepath: Path, field: str) -> list[str]:
    values: set[str] = set()
    for item in _iter_jsonl(filepath):
        v = item.get(field)
        if v:
            values.add(str(v))
    return sorted(values)
