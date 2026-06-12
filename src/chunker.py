from __future__ import annotations
import hashlib
from .models import Message, Chunk


def _chunk_id(level: int, thread_id: str, key: int) -> str:
    """Generate a stable 16-char hex ID for a chunk.

    The key is typically a timestamp-derived integer that, combined with the
    level and thread, makes the ID unique within the store.
    """
    raw = f"L{level}:{thread_id}:{key}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _split_by_gap(messages: list[Message], gap_minutes: float) -> list[list[Message]]:
    """Partition messages into groups separated by silences ≥ gap_minutes.

    Each returned group is a contiguous burst of conversation activity with no
    internal pause longer than the threshold. The sliding window in chunk_level2
    operates within each group independently so chunks never straddle a silence.
    """
    if not messages:
        return []
    groups: list[list[Message]] = []
    current: list[Message] = [messages[0]]
    gap_seconds = gap_minutes * 60
    for prev, curr in zip(messages, messages[1:]):
        if (curr.timestamp - prev.timestamp) >= gap_seconds:
            groups.append(current)
            current = [curr]
        else:
            current.append(curr)
    groups.append(current)
    return groups


def _fmt(messages: list[Message]) -> str:
    """Render a list of messages as a human-readable, LLM-friendly string.

    Each line is "[YYYY-MM-DD HH:MM] sender: text", matching the format
    cited in LLM answer prompts so the model can reference timestamps directly.
    """
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
    gap_minutes: float = 60.0,
) -> list[Chunk]:
    """Produce sliding-window conversation chunks (primary retrieval granularity).

    Messages are first partitioned into conversation groups separated by silences
    of at least *gap_minutes*. The sliding window then operates independently
    within each group, so no chunk ever spans an hourly gap between bursts.

    Within each group, consecutive windows share *overlap* messages to prevent a
    relevant exchange from being split across two non-overlapping chunks.

    Args:
        messages: Time-sorted list of messages for a single thread.
        thread_id: Conversation identifier used in chunk IDs and metadata.
        participants: All participants in the thread.
        window: Number of messages per chunk.
        overlap: Number of messages re-used at the start of each new chunk.
        gap_minutes: Silence threshold (minutes) that starts a new group.

    Returns:
        List of L2 Chunk objects in chronological order.
    """
    chunks: list[Chunk] = []
    step = window - overlap
    groups = _split_by_gap(messages, gap_minutes)

    for group in groups:
        for i in range(0, len(group), step):
            batch = group[i : i + window]
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
    """Produce topic segments inferred from temporal gaps (conversation sessions).

    A new segment begins whenever consecutive messages are separated by more
    than *gap_hours* hours, or when the current segment reaches *max_size*.
    Segments shorter than *min_size* messages are discarded — they are usually
    isolated reactions or single stray messages with little retrieval value.

    Args:
        messages: Time-sorted list of messages for a single thread.
        thread_id: Conversation identifier.
        participants: All participants in the thread.
        gap_hours: Silence duration (hours) that signals a new conversation session.
        min_size: Minimum messages required to keep a segment.
        max_size: Hard cap on segment length to prevent very long LLM contexts.

    Returns:
        List of L3 Chunk objects, one per inferred session.
    """
    if not messages:
        return []

    segments: list[list[Message]] = []
    current: list[Message] = [messages[0]]

    for prev, curr in zip(messages, messages[1:]):
        gap = (curr.timestamp - prev.timestamp) / 3600  # seconds → hours
        if gap >= gap_hours or len(current) >= max_size:
            if len(current) >= min_size:
                segments.append(current)
            current = [curr]
        else:
            current.append(curr)

    # Flush the final segment
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
    """Produce all L2 and L3 chunks for a thread in a single call.

    L1 (individual messages) is handled directly by MessageStore and is not
    returned here — it is never embedded into Chroma, only indexed by BM25.

    Returns:
        (l2_chunks, l3_chunks)
    """
    l2 = chunk_level2(messages, thread_id, participants)
    l3 = chunk_level3(messages, thread_id, participants)
    return l2, l3
