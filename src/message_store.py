from __future__ import annotations
import json
import pickle
from pathlib import Path
from .models import Message


class MessageStore:
    """
    In-memory store for all parsed messages.

    Supports:
    - Fast lookup by message ID
    - Context expansion (±N messages around a hit)
    - Iteration / slicing for BM25 ingestion
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._id_to_idx: dict[str, int] = {}

    def load(self, messages: list[Message]) -> None:
        self._messages = messages
        self._id_to_idx = {m.id: i for i, m in enumerate(messages)}

    def get_context(
        self,
        msg_id: str,
        before: int = 5,
        after: int = 5,
    ) -> list[Message]:
        """Return messages surrounding msg_id (inclusive)."""
        idx = self._id_to_idx.get(msg_id)
        if idx is None:
            return []
        start = max(0, idx - before)
        end = min(len(self._messages), idx + after + 1)
        return self._messages[start:end]

    def get_by_id(self, msg_id: str) -> Message | None:
        idx = self._id_to_idx.get(msg_id)
        return self._messages[idx] if idx is not None else None

    def get_slice(self, start_ts: int, end_ts: int) -> list[Message]:
        return [m for m in self._messages if start_ts <= m.timestamp <= end_ts]

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._messages, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_file(cls, path: str | Path) -> "MessageStore":
        with open(path, "rb") as f:
            messages: list[Message] = pickle.load(f)
        store = cls()
        store.load(messages)
        return store

    @property
    def messages(self) -> list[Message]:
        return self._messages
