from __future__ import annotations
import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from .models import Chunk

_LEVEL_COLLECTION = {2: "ig_messages_l2", 3: "ig_messages_l3"}


class ChromaStore:
    """Manages separate Chroma collections for L2 (windows) and L3 (topic segments)."""

    def __init__(self, persist_dir: str | Path, embeddings: Embeddings) -> None:
        self.persist_dir = Path(persist_dir)
        self.embeddings = embeddings
        self._stores: dict[int, Chroma] = {}

        for level, name in _LEVEL_COLLECTION.items():
            self._stores[level] = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=embeddings,
                collection_name=name,
            )

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        by_level: dict[int, list[Chunk]] = {2: [], 3: []}
        for c in chunks:
            if c.level in by_level:
                by_level[c.level].append(c)

        for level, level_chunks in by_level.items():
            if not level_chunks:
                continue

            docs, ids = [], []
            for c in level_chunks:
                docs.append(Document(page_content=c.text, metadata=_to_meta(c)))
                ids.append(c.chunk_id)

            for i in range(0, len(docs), batch_size):
                self._stores[level].add_documents(
                    docs[i : i + batch_size], ids=ids[i : i + batch_size]
                )

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        level: int,
        k: int = 30,
        where: dict | None = None,
    ) -> list[tuple[Document, float]]:
        return self._stores[level].similarity_search_with_score(
            query, k=k, filter=where
        )

    # ── Counts ─────────────────────────────────────────────────────────────────

    def count(self, level: int) -> int:
        return self._stores[level]._collection.count()

    def is_populated(self) -> bool:
        return self.count(2) > 0 and self.count(3) > 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_meta(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "level": c.level,
        "thread_id": c.thread_id,
        "participants": ", ".join(c.participants),
        "start_timestamp": c.start_timestamp,
        "end_timestamp": c.end_timestamp,
        "start_date": c.start_date,
        "end_date": c.end_date,
        "year": c.year,
        "month": c.month,
        "message_count": c.message_count,
        "sender": c.sender or "",
        "topic": c.topic or "",
    }


def doc_to_chunk(doc: Document) -> Chunk:
    m = doc.metadata
    return Chunk(
        chunk_id=m["chunk_id"],
        level=m["level"],
        thread_id=m["thread_id"],
        participants=m["participants"].split(", "),
        text=doc.page_content,
        message_ids=[],
        start_timestamp=m["start_timestamp"],
        end_timestamp=m["end_timestamp"],
        start_date=m["start_date"],
        end_date=m["end_date"],
        year=m["year"],
        month=m["month"],
        message_count=m["message_count"],
        sender=m.get("sender") or None,
        topic=m.get("topic") or None,
    )
