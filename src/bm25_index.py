from __future__ import annotations
import pickle
import re
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokenizer suitable for Spanish."""
    return re.findall(r"\w+", text.lower(), re.UNICODE)


class BM25Index:
    """BM25Okapi index over individual messages with persistence."""

    def __init__(self) -> None:
        self._index = None
        self._ids: list[str] = []

    def build(self, items: list[tuple[str, str]]) -> None:
        """items: list of (message_id, text)"""
        from rank_bm25 import BM25Okapi

        self._ids = [item[0] for item in items]
        tokenized = [_tokenize(item[1]) for item in items]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Returns list of (message_id, score) sorted descending."""
        if self._index is None:
            return []
        tokens = _tokenize(query)
        scores = self._index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._ids[i], float(s)) for i, s in ranked[:k] if s > 0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"ids": self._ids, "index": self._index}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._ids = data["ids"]
        obj._index = data["index"]
        return obj

    @property
    def size(self) -> int:
        return len(self._ids)
