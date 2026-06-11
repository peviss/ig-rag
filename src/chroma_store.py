from __future__ import annotations
import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from .models import Chunk

# One Chroma collection per chunk level so similarity searches can be scoped
# to a single granularity without post-filtering.
_LEVEL_COLLECTION = {2: "ig_messages_l2", 3: "ig_messages_l3"}


class ChromaStore:
    """Manages separate Chroma vector collections for L2 windows and L3 segments.

    L2 and L3 chunks are stored in different Chroma collections so that
    retrieval can weight them independently. Both collections live under the
    same persist directory on disk.
    """

    def __init__(self, persist_dir: str | Path, embeddings: Embeddings) -> None:
        """Initialise (or re-open) the Chroma collections.

        Chroma creates the collections on first use and reopens them on
        subsequent calls, so this constructor is safe to call multiple times
        against the same directory.

        Args:
            persist_dir: Directory where Chroma stores its SQLite + HNSW files.
            embeddings: Embedding function used for both indexing and querying.
        """
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
        """Embed and store chunks into their respective level collections.

        Chunks are batched to avoid overwhelming the embedding model's memory
        budget in a single call. Mixed-level inputs are split before dispatch.

        Args:
            chunks: L2 and/or L3 Chunk objects to embed and store.
            batch_size: Maximum documents per Chroma `add_documents` call.
        """
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
        """Run a similarity search against one level collection.

        Args:
            query: Raw query text (embedded internally).
            level: 2 or 3, selecting which collection to search.
            k: Maximum number of results.
            where: Optional Chroma metadata filter (ChromaDB `$and`/`$gte` syntax).

        Returns:
            List of (Document, distance) pairs, unsorted (Chroma returns them
            in ascending distance order, i.e. best match first).
        """
        return self._stores[level].similarity_search_with_score(
            query, k=k, filter=where
        )

    # ── Counts ─────────────────────────────────────────────────────────────────

    def count(self, level: int) -> int:
        """Return the number of documents stored in a level collection."""
        return self._stores[level]._collection.count()

    def is_populated(self) -> bool:
        """Return True if both L2 and L3 collections contain at least one document.

        Used during ingest to skip re-embedding when the index already exists
        and the caller has not passed force=True.
        """
        return self.count(2) > 0 and self.count(3) > 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_meta(c: Chunk) -> dict:
    """Convert a Chunk to a flat dict suitable for Chroma metadata storage.

    Chroma metadata values must be scalars (str, int, float, bool). Lists are
    joined as comma-separated strings; None fields are stored as empty strings
    so they are always present and filterable.
    """
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
    """Reconstruct a Chunk from a Chroma Document returned by a search.

    `message_ids` is not stored in Chroma metadata (it would be too large),
    so it is always restored as an empty list. Callers that need the original
    message IDs must look them up via MessageStore separately.
    """
    m = doc.metadata
    return Chunk(
        chunk_id=m["chunk_id"],
        level=m["level"],
        thread_id=m["thread_id"],
        # Reverse the join applied in _to_meta
        participants=m["participants"].split(", "),
        text=doc.page_content,
        message_ids=[],   # not persisted in Chroma — intentionally empty
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
