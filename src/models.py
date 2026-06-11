from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal


class Message(BaseModel):
    """A single parsed Instagram message from a DM thread.

    Timestamps are stored in both raw UNIX seconds and as a timezone-aware
    datetime so callers can choose whichever form is more convenient.
    """

    id: str                          # 16-char MD5 hex, stable across re-ingests
    thread_id: str                   # directory name of the conversation folder
    participants: list[str]          # all participants in the thread (mojibake-fixed)
    sender: str                      # display name of the message author
    text: str                        # normalised message body (may be a synthetic label for media)
    timestamp: int                   # UNIX seconds (UTC)
    dt: datetime                     # timezone-aware equivalent of timestamp
    has_media: bool = False
    media_type: Optional[str] = None         # "photo" | "video" | "audio" | "share"
    share_link: Optional[str] = None         # URL from a shared link, if present
    share_text: Optional[str] = None         # caption/description of a shared link
    reactions: list[dict] = Field(default_factory=list)  # [{"actor": ..., "reaction": ...}]


class Chunk(BaseModel):
    """A contiguous block of messages used as the retrieval unit in the RAG index.

    Three granularity levels are used:
      - L1: individual messages (held in MessageStore, not embedded)
      - L2: sliding-window batches of ~20 messages (primary dense retrieval)
      - L3: session segments split by temporal gaps (topic-level context)
    """

    chunk_id: str
    level: Literal[1, 2, 3]
    thread_id: str
    participants: list[str]
    text: str                      # pre-formatted "[date] sender: text" lines joined by newlines
    message_ids: list[str]
    start_timestamp: int           # UNIX seconds of the first message in the chunk
    end_timestamp: int             # UNIX seconds of the last message in the chunk
    start_date: str                # "YYYY-MM-DD" of start_timestamp
    end_date: str                  # "YYYY-MM-DD" of end_timestamp
    year: int
    month: int
    topic: Optional[str] = None    # reserved for future LLM-inferred topic labels
    message_count: int
    sender: Optional[str] = None   # L1 only — sender of the single message


class QueryIntent(BaseModel):
    """Structured interpretation of a natural-language query.

    Produced by `QueryEngine.understand()` either via rule-based heuristics
    (no API key) or an LLM call. Used to steer retrieval and re-ranking.
    """

    type: Literal["search", "summarize", "timeline", "count", "comparison"] = "search"
    entities: list[str] = Field(default_factory=list)   # named entities extracted from the query
    date_start: Optional[int] = None   # unix timestamp — filter chunks that end after this
    date_end: Optional[int] = None     # unix timestamp — filter chunks that start before this
    participants: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)  # synonym/paraphrase variants for recall
    temporal_anchor: Optional[Literal["first", "last", "before", "after", "during"]] = None


class RetrievalResult(BaseModel):
    """A single ranked chunk returned by the query engine, with its score and position."""

    chunk_id: str
    chunk: Chunk
    score: float   # combined RRF + reranker score
    rank: int      # 0-based position in the final result list


class AnswerResult(BaseModel):
    """Complete response from `QueryEngine.run()`, including retrieved sources."""

    answer: str                            # LLM-generated answer, or "" when no API key
    sources: list[RetrievalResult]
    query: str
    intent: Optional[QueryIntent] = None
    has_llm_answer: bool = True            # False when the engine runs without an Anthropic key
