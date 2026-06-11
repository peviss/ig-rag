from __future__ import annotations
import json
import pickle
from pathlib import Path
from .models import Message


class MessageStore:
    """In-memory store for all parsed messages with fast lookup and persistence.

    Holds the full corpus in a list (time-sorted) and a parallel dict for O(1)
    ID-based access. The two primary use-cases are:

    - BM25 ingestion: iterate over all messages to build the keyword index.
    - Context expansion: given a BM25 hit message ID, fetch the N messages
      before and after it to form a window chunk for the hybrid retriever.
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._id_to_idx: dict[str, int] = {}  # message_id → position in _messages

    def load(self, messages: list[Message]) -> None:
        """Populate the store from an already-parsed list of messages.

        Rebuilds the ID index on every call, so this replaces any previously
        loaded data rather than appending to it.

        Args:
            messages: Time-sorted Message objects (all threads merged).
        """
        self._messages = messages
        self._id_to_idx = {m.id: i for i, m in enumerate(messages)}

    def get_context(
        self,
        msg_id: str,
        before: int = 5,
        after: int = 5,
    ) -> list[Message]:
        """Return messages surrounding a given message ID (inclusive).

        Used by the hybrid retriever to expand a BM25 keyword hit into a
        window of surrounding context before fusing with dense search results.

        Args:
            msg_id: ID of the pivot message.
            before: Number of messages to include before the pivot.
            after: Number of messages to include after the pivot.

        Returns:
            Slice of the message list centred on the pivot, clamped to corpus
            boundaries. Returns [] if msg_id is not in the store.
        """
        idx = self._id_to_idx.get(msg_id)
        if idx is None:
            return []
        start = max(0, idx - before)
        end = min(len(self._messages), idx + after + 1)
        return self._messages[start:end]

    def get_by_id(self, msg_id: str) -> Message | None:
        """Look up a single message by its ID.

        Returns None if the ID is not present rather than raising KeyError.
        """
        idx = self._id_to_idx.get(msg_id)
        return self._messages[idx] if idx is not None else None

    def get_slice(self, start_ts: int, end_ts: int) -> list[Message]:
        """Return all messages whose timestamp falls within [start_ts, end_ts].

        Both bounds are inclusive. Timestamps are UNIX seconds.
        """
        return [m for m in self._messages if start_ts <= m.timestamp <= end_ts]

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle the message list to *path* for fast reloading on restart.

        The ID index is not pickled — it is cheap to rebuild from the list.

        Args:
            path: Destination file path (parent directories are created if absent).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._messages, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_file(cls, path: str | Path) -> "MessageStore":
        """Load a MessageStore from a file produced by `save()`.

        Args:
            path: Path to the pickle file.

        Returns:
            A fully initialised MessageStore ready for queries.
        """
        with open(path, "rb") as f:
            messages: list[Message] = pickle.load(f)
        store = cls()
        store.load(messages)
        return store

    @property
    def messages(self) -> list[Message]:
        """Read-only view of the full sorted message list."""
        return self._messages
