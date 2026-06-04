from __future__ import annotations
from langchain_core.embeddings import Embeddings

# Recommended models (best → fastest):
#   intfloat/multilingual-e5-large         — best Spanish retrieval, 1024d
#   BAAI/bge-m3                            — excellent, multilingual
#   paraphrase-multilingual-mpnet-base-v2  — good balance
#   paraphrase-multilingual-MiniLM-L12-v2  — fast default

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualEmbeddings(Embeddings):
    """
    Wraps sentence-transformers with E5-style prefix support.
    E5 models require "query: " / "passage: " prefixes for retrieval.
    """

    _PASSAGE_PREFIX = ""
    _QUERY_PREFIX = ""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

        if "e5" in model_name.lower():
            self._PASSAGE_PREFIX = "passage: "
            self._QUERY_PREFIX = "query: "

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._PASSAGE_PREFIX + t for t in texts]
        return self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            [self._QUERY_PREFIX + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

    @property
    def dimension(self) -> int:
        try:
            return self._model.get_embedding_dimension()
        except AttributeError:
            return self._model.get_sentence_embedding_dimension()
