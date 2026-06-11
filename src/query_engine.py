from __future__ import annotations
import json
import os
import re
from typing import TYPE_CHECKING

from .models import QueryIntent, Chunk, RetrievalResult, AnswerResult

if TYPE_CHECKING:
    from .retriever import HybridRetriever
    from .reranker import CrossEncoderReranker


# ── Rule-based fallback (no API key needed) ────────────────────────────────────

# Regex patterns used to classify query intent without an LLM call.
_INTENT_PATTERNS: dict[str, list[str]] = {
    "timeline": [r"\bcuándo\b", r"\bwhen\b", r"\bfecha\b", r"\bprimera vez\b", r"\bprimero\b", r"\bfirst\b"],
    "summarize": [r"\bresume\b", r"\bsummarize\b", r"\bresumen\b", r"\bde qué hablamos\b", r"\bsummary\b"],
    "count": [r"\bcuánta[s]?\b", r"\bhow many\b", r"\bcuánto[s]?\b"],
    "comparison": [r"\bdiferencia\b", r"\bcompara\b", r"\bcompare\b"],
}

# Keywords that indicate a temporal position relative to the conversation timeline.
_TEMPORAL_ANCHORS = {
    "first": ["primera vez", "primero", "primer", "first", "earliest", "antes de todo"],
    "last": ["última vez", "último", "last", "latest", "más reciente"],
    "before": ["antes", "before", "previo"],
    "after": ["después", "after", "luego"],
    "during": ["durante", "while", "en el período"],
}

# Topic-specific synonym expansions to improve recall for common query themes.
# Each entry maps a trigger keyword to a list of semantically related terms.
_QUERY_SYNONYMS: dict[str, list[str]] = {
    "mudarse": ["mudarse", "trasladarse", "vivir en", "moverse a", "relocate"],
    "españa": ["españa", "madrid", "barcelona", "español", "spain"],
    "trabajo": ["trabajo", "laboral", "empleo", "trabajar", "job"],
    "restaurante": ["restaurante", "comer", "cenar", "comida", "gastronomía"],
    "viaje": ["viaje", "viajar", "vacaciones", "trip", "vacacionar"],
    "regalo": ["regalo", "present", "regalito"],
    "casa": ["casa", "apartamento", "piso", "hogar"],
    "amor": ["amor", "querer", "te quiero", "te amo", "love"],
}


def _rule_intent(query: str) -> str:
    """Classify query intent using regex patterns, defaulting to 'search'."""
    q = query.lower()
    for intent, patterns in _INTENT_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            return intent
    return "search"


def _rule_temporal(query: str) -> str | None:
    """Extract a temporal anchor keyword from the query, or return None."""
    q = query.lower()
    for anchor, keywords in _TEMPORAL_ANCHORS.items():
        if any(kw in q for kw in keywords):
            return anchor
    return None


def _rule_expand(query: str) -> list[str]:
    """Generate up to 5 query variants by substituting domain synonyms.

    Only substitutes the first matching synonym per query to avoid creating
    semantically drifted variants. The original query is always first in the list.
    """
    queries = [query]
    q = query.lower()
    for key, synonyms in _QUERY_SYNONYMS.items():
        if key in q:
            for syn in synonyms:
                if syn != key and len(queries) < 5:
                    queries.append(re.sub(re.escape(key), syn, q, count=1))
    return queries


# ── LLM-based query understanding ─────────────────────────────────────────────

# Prompt used to extract structured intent from a query via the Claude API.
# The JSON-only instruction is critical — the parser uses a regex to extract
# the first JSON object from the response.
_UNDERSTAND_PROMPT = """\
Analiza esta pregunta sobre una conversación de Instagram y responde SOLO en JSON válido.

Pregunta: "{query}"

Formato de respuesta (sin texto adicional):
{{
  "intent": "search" | "summarize" | "timeline" | "count" | "comparison",
  "entities": ["lista", "de", "entidades"],
  "temporal_anchor": null | "first" | "last" | "before" | "after" | "during",
  "date_start": null | "YYYY-MM-DD",
  "date_end": null | "YYYY-MM-DD",
  "expanded_queries": ["consulta1", "consulta2", "consulta3", "consulta4"]
}}

Genera expanded_queries en el mismo idioma que la pregunta original.
"""

# System prompt for the answer-generation call. The citation format [Fuente N | date]
# must match the context block format built in `_answer` so the model can reference
# the correct source numbers.
_ANSWER_SYSTEM = """\
Eres un asistente especializado en analizar conversaciones de Instagram.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con información de los mensajes proporcionados.
2. Cita siempre la fuente con su fecha: [Fuente N | YYYY-MM-DD].
3. Para preguntas temporales (primera/última vez), ordena cronológicamente y cita la fecha exacta del primer/último mensaje relevante.
4. Si no encuentras evidencia, di exactamente: "No encontré evidencia en los mensajes exportados."
5. NUNCA inventes fechas, nombres ni hechos.
6. Sé conciso pero completo. Usa viñetas cuando sea útil.
"""

_ANSWER_USER = """\
Pregunta: {query}

Mensajes recuperados:
{context}

Responde la pregunta citando las fuentes con sus fechas.\
"""


class QueryEngine:
    """End-to-end RAG pipeline: understand → retrieve → rerank → answer.

    When an ANTHROPIC_API_KEY is present, intent classification and final
    answer generation use Claude. Without a key, rule-based heuristics are used
    for intent and retrieval still runs normally — the answer field is just
    left empty (`has_llm_answer=False`).
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        """Initialise the engine and optionally connect to the Anthropic API.

        Args:
            retriever: Configured HybridRetriever instance.
            reranker: Optional cross-encoder. When None, retrieval results are
                      used as-is (truncated to top_k).
            model: Claude model ID used for both intent parsing and answering.
        """
        self.retriever = retriever
        self.reranker = reranker
        self.model = model
        self._client = None  # None when no API key is configured

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, query: str, top_k: int = 8) -> AnswerResult:
        """Execute the full RAG pipeline and return a cited answer.

        Stages: understand → retrieve (hybrid) → rerank → answer (LLM or empty).

        Args:
            query: Raw user query string.
            top_k: Number of chunks passed to the LLM answer prompt.

        Returns:
            AnswerResult with an answer string and ranked source chunks.
        """
        intent = self.understand(query)
        candidates = self.retrieve(query, intent, k_candidates=40)
        top_chunks = self._rerank(query, candidates, top_k=top_k)
        return self._answer(query, intent, top_chunks)

    def understand(self, query: str) -> QueryIntent:
        """Parse a query into a structured QueryIntent.

        Uses the LLM when a client is available, otherwise falls back to
        rule-based classification and synonym expansion.
        """
        if self._client:
            return self._llm_understand(query)
        return QueryIntent(
            type=_rule_intent(query),
            temporal_anchor=_rule_temporal(query),
            expanded_queries=_rule_expand(query),
        )

    def retrieve(
        self,
        query: str,
        intent: QueryIntent,
        k_candidates: int = 40,
    ) -> list[tuple[Chunk, float]]:
        """Run hybrid retrieval using the expanded query list from intent.

        For temporal queries ("first"/"last"), results are re-sorted by
        timestamp after RRF fusion so the chronologically earliest or latest
        chunks surface at the top regardless of relevance score.

        Args:
            query: Original query (used as fallback if intent has no expansions).
            intent: Structured intent including expanded queries and date bounds.
            k_candidates: Maximum chunks returned before re-ranking.

        Returns:
            List of (chunk, score) pairs, potentially time-sorted.
        """
        queries = intent.expanded_queries or [query]

        candidates = self.retriever.retrieve(
            queries=queries,
            k_dense=40,
            k_bm25=50,
            k_final=k_candidates,
            start_ts=intent.date_start,
            end_ts=intent.date_end,
        )

        # Temporal re-sorting: for "first"/"last" queries sort by time, not by score.
        if intent.temporal_anchor in ("first", "earliest"):
            candidates.sort(key=lambda x: x[0].start_timestamp)
        elif intent.temporal_anchor in ("last", "latest"):
            candidates.sort(key=lambda x: x[0].start_timestamp, reverse=True)

        return candidates

    # ── Private helpers ────────────────────────────────────────────────────────

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Apply the cross-encoder reranker if configured, otherwise truncate.

        Args:
            query: Original query string.
            candidates: RRF-fused retrieval results.
            top_k: Desired number of final chunks.

        Returns:
            Top-k chunks by reranker score, or top-k by RRF score if no reranker.
        """
        if self.reranker and candidates:
            return self.reranker.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]

    def _llm_understand(self, query: str) -> QueryIntent:
        """Call the Claude API to extract structured intent from a query.

        The model is instructed to respond with a JSON object only. A regex
        extracts the first `{...}` block from the response as a safety measure
        against models that add prose before or after the JSON. Falls back to
        a plain "search" intent on any parse failure.

        Date strings from the LLM ("YYYY-MM-DD") are converted to UNIX
        timestamps for use in Chroma metadata filters.
        """
        from datetime import datetime

        prompt = _UNDERSTAND_PROMPT.format(query=query)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Extract the first JSON object in case the model adds surrounding text.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return QueryIntent(type="search", expanded_queries=[query])
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return QueryIntent(type="search", expanded_queries=[query])

        # Convert YYYY-MM-DD strings to UNIX timestamps; ignore malformed dates.
        start_ts = end_ts = None
        if data.get("date_start"):
            try:
                start_ts = int(datetime.strptime(data["date_start"], "%Y-%m-%d").timestamp())
            except ValueError:
                pass
        if data.get("date_end"):
            try:
                end_ts = int(datetime.strptime(data["date_end"], "%Y-%m-%d").timestamp())
            except ValueError:
                pass

        return QueryIntent(
            type=data.get("intent", "search"),
            entities=data.get("entities", []),
            temporal_anchor=data.get("temporal_anchor"),
            date_start=start_ts,
            date_end=end_ts,
            # Prepend the original query so it always has the highest weight
            # in the hybrid retriever's multi-query fusion.
            expanded_queries=[query] + data.get("expanded_queries", []),
        )

    def _answer(
        self,
        query: str,
        intent: QueryIntent,
        chunks: list[tuple[Chunk, float]],
    ) -> AnswerResult:
        """Generate a cited answer from the top-k chunks using the Claude API.

        Each chunk is labelled "[Fuente N | start_date → end_date]" so the model
        can reference specific sources in its answer. When no API client is
        configured, returns an empty answer with `has_llm_answer=False` so the
        frontend can display raw sources instead.

        Args:
            query: Original user query.
            intent: Parsed intent (attached to the result for API consumers).
            chunks: Reranked (chunk, score) pairs to include as context.

        Returns:
            AnswerResult with a cited answer string and source list.
        """
        sources = [
            RetrievalResult(chunk_id=c.chunk_id, chunk=c, score=s, rank=i)
            for i, (c, s) in enumerate(chunks)
        ]

        if not self._client:
            return AnswerResult(
                answer="",
                sources=sources,
                query=query,
                intent=intent,
                has_llm_answer=False,
            )

        # Build the context block — source numbers must match [Fuente N] citations.
        context_parts = []
        for i, (chunk, _) in enumerate(chunks, 1):
            context_parts.append(
                f"[Fuente {i} | {chunk.start_date} → {chunk.end_date}]\n{chunk.text}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        user_msg = _ANSWER_USER.format(query=query, context=context_str)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=_ANSWER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        return AnswerResult(
            answer=response.content[0].text,
            sources=sources,
            query=query,
            intent=intent,
        )
