from __future__ import annotations
from langchain_core.embeddings import Embeddings

# Recommended models ranked best → fastest for Spanish/multilingual retrieval:
#   intfloat/multilingual-e5-large         — best Spanish retrieval, 1024d
#   BAAI/bge-m3                            — excellent multilingual coverage
#   paraphrase-multilingual-mpnet-base-v2  — good balance of quality and speed
#   paraphrase-multilingual-MiniLM-L12-v2  — fast default used in production

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualEmbeddings(Embeddings):
    """Sentence-transformers wrapper with E5-style query/passage prefix support.

    E5 models (intfloat/multilingual-e5-*) require "query: " and "passage: "
    prefixes to distinguish retrieval mode from indexing mode. This class
    detects the model family by name and applies the correct prefixes
    automatically, so callers never need to handle it themselves.

    Non-E5 models receive empty prefixes and behave as plain sentence encoders.
    """

    _PASSAGE_PREFIX = ""
    _QUERY_PREFIX = ""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        """Load the sentence-transformer model and configure E5 prefixes if needed.

        Args:
            model_name: HuggingFace model identifier or local path.
            device: PyTorch device string ("cpu", "cuda", "mps", …).
            batch_size: Number of texts encoded per forward pass during ingestion.
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

        # E5 models must receive asymmetric prefixes — "passage: " at index time
        # and "query: " at search time — to match how they were fine-tuned.
        if "e5" in model_name.lower():
            self._PASSAGE_PREFIX = "passage: "
            self._QUERY_PREFIX = "query: "

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of passages for storage in the vector index.

        Normalised embeddings are used so cosine similarity equals dot product,
        which is what Chroma's default distance metric expects.

        Args:
            texts: Raw passage strings (no prefix — this method adds it).

        Returns:
            List of float vectors, one per input text.
        """
        prefixed = [self._PASSAGE_PREFIX + t for t in texts]
        return self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for similarity search.

        Progress bar is suppressed here since queries are encoded one at a time
        during interactive use where bar output would be noise.

        Args:
            text: The user's query (no prefix — this method adds it).

        Returns:
            A single normalised float vector.
        """
        return self._model.encode(
            [self._QUERY_PREFIX + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

    @property
    def dimension(self) -> int:
        """Output vector dimension for the loaded model."""
        try:
            return self._model.get_embedding_dimension()
        except AttributeError:
            # Older sentence-transformers versions expose a different method name.
            return self._model.get_sentence_embedding_dimension()
