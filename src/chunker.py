from __future__ import annotations
import hashlib
from .models import Message, Chunk


def _chunk_id(level: int, thread_id: str, key: int) -> str:
    raw = f"L{level}:{thread_id}:{key}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _fmt(messages: list[Message]) -> str:
    lines = [
        f"[{m.dt.strftime('%Y-%m-%d %H:%M')}] {m.sender}: {m.text}"
        for m in messages
    ]
    return "\n".join(lines)


def chunk_level2(
    messages: list[Message],
    thread_id: str,
    participants: list[str],
    window: int = 20,
    overlap: int = 5,
) -> list[Chunk]:
    """Sliding-window conversation chunks (primary retrieval granularity)."""
    chunks: list[Chunk] = []
    step = window - overlap

    for i in range(0, len(messages), step):
        batch = messages[i : i + window]
        if not batch:
            break

        start, end = batch[0], batch[-1]
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(2, thread_id, start.timestamp * 1000 + i),
                level=2,
                thread_id=thread_id,
                participants=participants,
                text=_fmt(batch),
                message_ids=[m.id for m in batch],
                start_timestamp=start.timestamp,
                end_timestamp=end.timestamp,
                start_date=start.dt.strftime("%Y-%m-%d"),
                end_date=end.dt.strftime("%Y-%m-%d"),
                year=start.dt.year,
                month=start.dt.month,
                message_count=len(batch),
            )
        )
    return chunks


def chunk_level3(
    messages: list[Message],
    thread_id: str,
    participants: list[str],
    gap_hours: float = 4.0,
    min_size: int = 5,
    max_size: int = 200,
) -> list[Chunk]:
    """Topic segments inferred from temporal gaps (conversation sessions)."""
    if not messages:
        return []

    segments: list[list[Message]] = []
    current: list[Message] = [messages[0]]

    for prev, curr in zip(messages, messages[1:]):
        gap = (curr.timestamp - prev.timestamp) / 3600
        if gap >= gap_hours or len(current) >= max_size:
            if len(current) >= min_size:
                segments.append(current)
            current = [curr]
        else:
            current.append(curr)

    if len(current) >= min_size:
        segments.append(current)

    chunks: list[Chunk] = []
    for seg in segments:
        start, end = seg[0], seg[-1]
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(3, thread_id, start.timestamp),
                level=3,
                thread_id=thread_id,
                participants=participants,
                text=_fmt(seg),
                message_ids=[m.id for m in seg],
                start_timestamp=start.timestamp,
                end_timestamp=end.timestamp,
                start_date=start.dt.strftime("%Y-%m-%d"),
                end_date=end.dt.strftime("%Y-%m-%d"),
                year=start.dt.year,
                month=start.dt.month,
                message_count=len(seg),
            )
        )
    return chunks


def chunk_all(
    messages: list[Message],
    thread_id: str,
    participants: list[str],
) -> tuple[list[Chunk], list[Chunk]]:
    """Return (l2_chunks, l3_chunks). L1 is handled by MessageStore directly."""
    l2 = chunk_level2(messages, thread_id, participants)
    l3 = chunk_level3(messages, thread_id, participants)
    return l2, l3
