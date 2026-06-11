from __future__ import annotations
import pickle
import re
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokenizer suitable for Spanish.

    Uses `\w+` so accented characters (á, é, ñ, ü …) are treated as word
    characters rather than delimiters, which is essential for Spanish recall.
    """
    return re.findall(r"\w+", text.lower(), re.UNICODE)


class BM25Index:
    """BM25Okapi index over individual messages with file-based persistence.

    BM25 provides keyword recall that dense embeddings miss — exact names,
    slang, and rare terms that don't cluster well in embedding space. In the
    hybrid retriever, BM25 hits are expanded to a context window of surrounding
    messages before being fused with Chroma results.
    """

    def __init__(self) -> None:
        self._index = None           # rank_bm25.BM25Okapi instance, None until built
        self._ids: list[str] = []    # message IDs in the same order as the BM25 corpus

    def build(self, items: list[tuple[str, str]]) -> None:
        """Build the BM25 index from (message_id, text) pairs.

        Args:
            items: List of (message_id, raw text) tuples covering every message
                   in the store. Order determines the internal corpus index.
        """
        from rank_bm25 import BM25Okapi

        self._ids = [item[0] for item in items]
        tokenized = [_tokenize(item[1]) for item in items]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Return the top-k matching message IDs with their BM25 scores.

        Messages with a score of 0 are excluded — they have no term overlap
        with the query and would just add noise to the hybrid fusion step.

        Args:
            query: Raw query string (tokenised internally).
            k: Maximum number of results to return.

        Returns:
            List of (message_id, score) sorted by score descending.
        """
        if self._index is None:
            return []
        tokens = _tokenize(query)
        scores = self._index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._ids[i], float(s)) for i, s in ranked[:k] if s > 0]

    def save(self, path: str | Path) -> None:
        """Pickle the index and ID list to *path* for later reuse.

        Uses the highest available pickle protocol for compact output and fast
        deserialisation on the same Python version.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"ids": self._ids, "index": self._index}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Restore a previously saved BM25Index from *path*.

        Args:
            path: Path to a file produced by `save()`.

        Returns:
            A ready-to-use BM25Index instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._ids = data["ids"]
        obj._index = data["index"]
        return obj

    @property
    def size(self) -> int:
        """Number of messages currently indexed."""
        return len(self._ids)
