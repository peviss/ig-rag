from __future__ import annotations
from .models import Chunk

# Good multilingual cross-encoders for Spanish:
#   cross-encoder/mmarco-mMiniLMv2-L12-H384-v1  — multilingual, fast
#   cross-encoder/ms-marco-MiniLM-L-6-v2          — English only, fastest
DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
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
