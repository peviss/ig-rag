from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from .models import Chunk, Message
from .chroma_store import ChromaStore, doc_to_chunk
from .bm25_index import BM25Index
from .message_store import MessageStore


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """
    Reciprocal Rank Fusion over:
      - Dense vector search (L2 windows + L3 topic segments)
      - BM25 keyword search (individual messages, expanded to context windows)

    Retrieval weights: dense 60%, BM25 30%.
    """

    def __init__(
        self,
        chroma: ChromaStore,
        bm25: BM25Index,
        msg_store: MessageStore,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.3,
        context_window: int = 8,  # messages around a BM25 hit
    ) -> None:
        self.chroma = chroma
        self.bm25 = bm25
        self.msg_store = msg_store
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.context_window = context_window

    def retrieve(
        self,
        queries: list[str],
        k_dense: int = 40,
        k_bm25: int = 50,
        k_final: int = 20,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        scores: dict[str, float] = defaultdict(float)
        chunks_by_id: dict[str, Chunk] = {}

        chroma_filter = _build_filter(start_ts, end_ts)

        for query in queries:
            # ── Dense retrieval (L2 + L3) ──────────────────────────────────
            for level in (2, 3):
                results = self.chroma.search(query, level=level, k=k_dense, where=chroma_filter)
                for rank, (doc, _dist) in enumerate(results):
                    chunk = doc_to_chunk(doc)
                    cid = chunk.chunk_id
                    chunks_by_id[cid] = chunk
                    scores[cid] += self.dense_weight * _rrf(rank)

            # ── BM25 retrieval (L1 messages → expand to window) ────────────
            bm25_hits = self.bm25.search(query, k=k_bm25)
            for rank, (msg_id, _score) in enumerate(bm25_hits):
                context_msgs = self.msg_store.get_context(
                    msg_id, before=self.context_window, after=self.context_window
                )
                if not context_msgs:
                    continue
                if start_ts and context_msgs[-1].timestamp < start_ts:
                    continue
                if end_ts and context_msgs[0].timestamp > end_ts:
                    continue

                cid = f"bm25:{msg_id}"
                if cid not in chunks_by_id:
                    chunks_by_id[cid] = _msgs_to_chunk(context_msgs, msg_id)
                scores[cid] += self.bm25_weight * _rrf(rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (chunks_by_id[cid], score)
            for cid, score in ranked[:k_final]
            if cid in chunks_by_id
        ]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_filter(start_ts: int | None, end_ts: int | None) -> dict | None:
    if start_ts is None and end_ts is None:
        return None
    clauses = []
    if start_ts:
        clauses.append({"end_timestamp": {"$gte": start_ts}})
    if end_ts:
        clauses.append({"start_timestamp": {"$lte": end_ts}})
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _msgs_to_chunk(messages: list[Message], pivot_id: str) -> Chunk:
    from .chunker import _fmt
    start, end = messages[0], messages[-1]
    return Chunk(
        chunk_id=f"bm25:{pivot_id}",
        level=1,
        thread_id=start.thread_id,
        participants=start.participants,
        text=_fmt(messages),
        message_ids=[m.id for m in messages],
        start_timestamp=start.timestamp,
        end_timestamp=end.timestamp,
        start_date=start.dt.strftime("%Y-%m-%d"),
        end_date=end.dt.strftime("%Y-%m-%d"),
        year=start.dt.year,
        month=start.dt.month,
        message_count=len(messages),
    )
