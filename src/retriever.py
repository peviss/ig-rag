from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from .models import Chunk, Message
from .chroma_store import ChromaStore, doc_to_chunk
from .bm25_index import BM25Index
from .message_store import MessageStore


def _rrf(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a single result at position *rank*.

    RRF is rank-position-only — it ignores raw similarity scores — which makes
    it robust when fusing results from sources with incompatible score scales
    (e.g. cosine distance vs. BM25 IDF scores). The constant k=60 is the
    standard from the original RRF paper (Cormack et al., 2009).
    """
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """Reciprocal Rank Fusion over dense vector search and BM25 keyword search.

    Retrieval strategy:
      1. Dense (60% weight): query both the L2 (sliding window) and L3 (session)
         Chroma collections. Running both levels improves recall for queries that
         match at different granularities.
      2. BM25 (30% weight): find individual messages by keyword, then expand each
         hit to a context window of surrounding messages to form a pseudo-chunk.
      3. RRF fusion: accumulate weighted RRF scores across all queries and both
         sources, then return the top-k by combined score.

    Weights sum to less than 1.0 deliberately — the slack absorbs ties and
    keeps scores numerically stable when one source returns no results.
    """

    def __init__(
        self,
        chroma: ChromaStore,
        bm25: BM25Index,
        msg_store: MessageStore,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.3,
        context_window: int = 8,  # messages before/after each BM25 hit
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
        """Run hybrid retrieval for one or more query variants.

        Multiple query variants (from intent expansion) are each run through
        both sources independently; their RRF scores are summed so that chunks
        matching several query phrasings rank higher than those matching only one.

        Args:
            queries: List of query strings (original + semantic expansions).
            k_dense: Results to fetch per level per query from Chroma.
            k_bm25: Results to fetch per query from BM25.
            k_final: Maximum chunks to return after fusion.
            start_ts: Optional UNIX timestamp lower bound for chunk filtering.
            end_ts: Optional UNIX timestamp upper bound for chunk filtering.

        Returns:
            List of (Chunk, fused_score) sorted by score descending, at most k_final items.
        """
        scores: dict[str, float] = defaultdict(float)
        chunks_by_id: dict[str, Chunk] = {}

        chroma_filter = _build_filter(start_ts, end_ts)

        for query in queries:
            # ── Dense retrieval (L2 + L3) ──────────────────────────────────────
            for level in (2, 3):
                results = self.chroma.search(query, level=level, k=k_dense, where=chroma_filter)
                for rank, (doc, _dist) in enumerate(results):
                    chunk = doc_to_chunk(doc)
                    cid = chunk.chunk_id
                    chunks_by_id[cid] = chunk
                    scores[cid] += self.dense_weight * _rrf(rank)

            # ── BM25 retrieval (individual messages → expand to context window) ─
            bm25_hits = self.bm25.search(query, k=k_bm25)
            for rank, (msg_id, _score) in enumerate(bm25_hits):
                context_msgs = self.msg_store.get_context(
                    msg_id, before=self.context_window, after=self.context_window
                )
                if not context_msgs:
                    continue

                # Date-filter BM25 windows by checking whether the window
                # overlaps the requested range at all.
                if start_ts and context_msgs[-1].timestamp < start_ts:
                    continue
                if end_ts and context_msgs[0].timestamp > end_ts:
                    continue

                # Prefix the chunk ID to avoid collisions with Chroma chunk IDs.
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
    """Build a Chroma metadata filter that selects chunks overlapping a date range.

    A chunk overlaps the range [start_ts, end_ts] if its end is after start_ts
    AND its start is before end_ts. Neither bound is required.

    Returns None when no filtering is needed, which avoids sending an empty
    filter dict that Chroma would reject.
    """
    if start_ts is None and end_ts is None:
        return None
    clauses = []
    if start_ts:
        clauses.append({"end_timestamp": {"$gte": start_ts}})
    if end_ts:
        clauses.append({"start_timestamp": {"$lte": end_ts}})
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _msgs_to_chunk(messages: list[Message], pivot_id: str) -> Chunk:
    """Wrap a BM25 context window into a synthetic L1 Chunk for fusion.

    The resulting chunk uses level=1 and a "bm25:" prefix on the chunk_id to
    distinguish it from L2/L3 chunks stored in Chroma. It is never persisted —
    it exists only for the duration of a single retrieval call.

    Args:
        messages: Context window of messages surrounding a BM25 hit.
        pivot_id: ID of the BM25 hit message (used in the chunk_id).

    Returns:
        A transient Chunk representing the context window.
    """
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
