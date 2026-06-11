from __future__ import annotations
from .models import Chunk

# Recommended multilingual cross-encoders for Spanish re-ranking:
#   cross-encoder/mmarco-mMiniLMv2-L12-H384-v1  — multilingual, fast, good quality
#   cross-encoder/ms-marco-MiniLM-L-6-v2          — English only, fastest option

DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderReranker:
    """Cross-encoder re-ranker that scores (query, chunk) pairs jointly.

    Cross-encoders attend to both the query and the passage simultaneously,
    which gives significantly better relevance judgements than the cosine
    similarity used by bi-encoders. The trade-off is that they are too slow
    for first-stage retrieval — they are only applied to the top candidates
    returned by the hybrid retriever.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        """Load the cross-encoder model.

        Args:
            model_name: HuggingFace model identifier or local path.
        """
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int = 10,
        max_text_len: int = 512,
    ) -> list[tuple[Chunk, float]]:
        """Score each candidate against the query and return the top-k.

        Chunk text is truncated to *max_text_len* characters before being fed
        to the cross-encoder. This keeps inference fast and prevents out-of-memory
        errors on very large L3 segments while still capturing the most relevant
        content near the start of each chunk.

        Args:
            query: The user's original query string.
            candidates: (chunk, hybrid_score) pairs from the retriever, in any order.
            top_k: Number of chunks to return after re-ranking.
            max_text_len: Maximum characters of chunk text passed to the cross-encoder.

        Returns:
            List of (chunk, cross_encoder_score) sorted by score descending,
            at most top_k items.
        """
        if not candidates:
            return []

        pairs = [(query, c.text[:max_text_len]) for c, _ in candidates]
        raw_scores = self._model.predict(pairs)

        reranked = sorted(
            zip(candidates, raw_scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        return [(chunk, float(score)) for (chunk, _), score in reranked[:top_k]]
