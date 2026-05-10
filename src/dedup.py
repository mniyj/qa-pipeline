import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _ngrams(text: str, n: int = 3) -> set[int]:
    """Compute n-gram hash set for a text."""
    if not text or len(text) < n:
        return {hash(text)}
    return {hash(text[i:i + n]) for i in range(len(text) - n + 1)}


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _max_similarity(needle: set[int], haystack: list[set[int]]) -> float:
    best = 0.0
    for existing in haystack:
        sim = _jaccard(needle, existing)
        if sim > best:
            best = sim
        if best >= 0.99:
            break
    return best


class DedupTracker:
    """
    Lightweight dedup tracker using char n-gram Jaccard similarity.
    
    Loads fingerprints from existing output files so that before calling
    the LLM, you can check if the input content is too similar to already-
    processed content — and skip the API call.
    """

    def __init__(self, threshold: float = 0.70, ngram_size: int = 3):
        self.threshold = threshold
        self.ngram_size = ngram_size
        self._fingerprints: list[set[int]] = []
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def load_from_file(self, filepath: str | Path, field: str = "question"):
        """Load fingerprints from an existing jsonl file by extracting a text field."""
        fp = Path(filepath)
        if not fp.exists():
            return
        loaded = 0
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        text = item.get(field, "")
                        if text:
                            self._fingerprints.append(_ngrams(text, self.ngram_size))
                            loaded += 1
                    except json.JSONDecodeError:
                        continue
        self._count += loaded
        logger.info(f"DedupTracker 从 {fp.name} 加载 {loaded} 条指纹（共 {self._count} 条）")

    def load_from_dir(self, dirpath: str | Path, glob_pattern: str = "*.jsonl",
                      field: str = "question", exclude_report: bool = True):
        """Load fingerprints from all matching jsonl files in a directory."""
        dp = Path(dirpath)
        if not dp.exists():
            return
        for fp in sorted(dp.glob(glob_pattern)):
            if exclude_report and fp.name.endswith("_report.json"):
                continue
            if fp.name.startswith("."):
                continue
            self.load_from_file(fp, field=field)

    def add(self, text: str):
        """Register a new text's fingerprint (call after successful generation)."""
        if text:
            self._fingerprints.append(_ngrams(text, self.ngram_size))
            self._count += 1

    def deduped_add(self, text: str) -> bool:
        """
        Check if text is too similar to existing fingerprints, and if not, add it.
        Returns True if text was added (not duplicate), False if skipped as duplicate.
        """
        if not text:
            return False
        fp = _ngrams(text, self.ngram_size)
        if self._fingerprints and _max_similarity(fp, self._fingerprints) > self.threshold:
            return False
        self._fingerprints.append(fp)
        self._count += 1
        return True

    def is_duplicate(self, text: str) -> bool:
        """Check if text is too similar to any existing fingerprint (without adding)."""
        if not text or not self._fingerprints:
            return False
        fp = _ngrams(text, self.ngram_size)
        return _max_similarity(fp, self._fingerprints) > self.threshold
