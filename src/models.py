from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal


class Message(BaseModel):
    id: str
    thread_id: str
    participants: list[str]
    sender: str
    text: str
    timestamp: int  # unix seconds
    dt: datetime
    has_media: bool = False
    media_type: Optional[str] = None
    share_link: Optional[str] = None
    share_text: Optional[str] = None
    reactions: list[dict] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    level: Literal[1, 2, 3]
    thread_id: str
    participants: list[str]
    text: str
    message_ids: list[str]
    start_timestamp: int
    end_timestamp: int
    start_date: str
    end_date: str
    year: int
    month: int
    topic: Optional[str] = None
    message_count: int
    sender: Optional[str] = None  # L1 only


class QueryIntent(BaseModel):
    type: Literal["search", "summarize", "timeline", "count", "comparison"] = "search"
    entities: list[str] = Field(default_factory=list)
    date_start: Optional[int] = None  # unix timestamp
    date_end: Optional[int] = None    # unix timestamp
    participants: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)
    temporal_anchor: Optional[Literal["first", "last", "before", "after", "during"]] = None


class RetrievalResult(BaseModel):
    chunk_id: str
    chunk: Chunk
    score: float
    rank: int


class AnswerResult(BaseModel):
    answer: str
    sources: list[RetrievalResult]
    query: str
    intent: Optional[QueryIntent] = None
    has_llm_answer: bool = True
