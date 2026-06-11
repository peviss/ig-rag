from __future__ import annotations
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from .models import Message


# Instagram exports cap share_text at around 200 chars in practice; we mirror
# that ceiling so stored content stays predictable in size.
_MAX_SHARE_CHARS = 200


def _truncate(text: str, limit: int = _MAX_SHARE_CHARS) -> str:
    """Trim *text* to *limit* characters, appending '…' when truncated."""
    if not text or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def fix_encoding(text: str) -> str:
    """Fix Instagram's mojibake: bytes stored as Latin-1 chars re-encoded as UTF-8.

    Instagram's export pipeline serialises UTF-8 byte sequences as if each byte
    were an independent Latin-1 character, producing garbage like 'Ã³' instead
    of 'ó'. Re-encoding through Latin-1 → UTF-8 recovers the original string.
    Falls back to the original text if the round-trip fails (e.g. already valid
    ASCII, or truly mixed encoding).
    """
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _msg_id(thread_id: str, ts_ms: int, sender: str, idx: int) -> str:
    """Generate a stable 16-char hex ID for a single message.

    The key incorporates the thread, millisecond timestamp, sender name, and
    per-file index so that two messages sent at the same millisecond by
    different senders still get distinct IDs. MD5 is used purely for its
    speed and compact output — not for security.
    """
    key = f"{thread_id}:{ts_ms}:{sender}:{idx}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _parse_file(path: Path, thread_id: str) -> tuple[list[str], list[Message]]:
    """Parse a single Instagram message_N.json file into a list of Messages.

    Instagram splits long threads across multiple numbered files (message_1.json,
    message_2.json, …). This function handles one file at a time; callers are
    responsible for merging and sorting across files.

    Returns a (participants, messages) tuple. Messages that carry no text and no
    recognised media attachment are silently dropped — these are usually system
    events (e.g. 'liked a message') that have no useful content for the RAG index.

    Args:
        path: Absolute path to a message_N.json export file.
        thread_id: Directory name of the conversation (used in IDs and metadata).

    Returns:
        Tuple of (participant name list, parsed Message objects).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    participants = [fix_encoding(p["name"]) for p in data.get("participants", [])]
    messages: list[Message] = []

    for idx, raw in enumerate(data.get("messages", [])):
        sender = fix_encoding(raw.get("sender_name", "unknown"))
        ts_ms = raw.get("timestamp_ms", 0)
        ts = ts_ms // 1000  # convert milliseconds → seconds for standard UNIX timestamp
        content = fix_encoding(raw.get("content", "") or "")

        # Detect the media type present in this message. Only one type is
        # expected per message, so the first match wins.
        media_type: str | None = None
        share_link: str | None = None
        share_text: str | None = None
        has_media = False

        if raw.get("photos"):
            has_media, media_type = True, "photo"
        elif raw.get("videos"):
            has_media, media_type = True, "video"
        elif raw.get("audio_files"):
            has_media, media_type = True, "audio"
        elif raw.get("share"):
            has_media, media_type = True, "share"
            share = raw["share"]
            share_link = share.get("link", "")
            share_text = fix_encoding(share.get("share_text", "") or "")
            # When a share has no accompanying text, synthesise a human-readable
            # placeholder so the message isn't silently empty in the index.
            if not content:
                snippet = _truncate(share_text)
                content = f"[Compartido] {snippet}" if snippet else "[Enlace compartido]"

        # Normalise emoji reactions — both actor and reaction strings can be mojibaked.
        reactions = [
            {
                "actor": fix_encoding(r.get("actor", "")),
                "reaction": fix_encoding(r.get("reaction", "")),
            }
            for r in raw.get("reactions", [])
        ]

        # Skip messages with no searchable content (e.g. pure like/heart events).
        if not content and not has_media:
            continue
        # Fallback label when media is present but the content field is empty.
        if not content:
            content = f"[{media_type}]"

        # Replace auto-generated attachment placeholders with the real share text.
        # Instagram sometimes fills `content` with a generic "sent an attachment"
        # string instead of the actual share_text; we overwrite those here.
        _attachment_phrases = (
            "adjunto",      # covers all "enviado un archivo adjunto" variants
            "attachment",   # English export variant
        )
        if any(phrase in content.lower() for phrase in _attachment_phrases):
            if share_text:
                content = f"[Compartido] {_truncate(share_text)}"
            elif share_link:
                content = f"[Enlace] {share_link}"

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        messages.append(
            Message(
                id=_msg_id(thread_id, ts_ms, sender, idx),
                thread_id=thread_id,
                participants=participants,
                sender=sender,
                text=content,
                timestamp=ts,
                dt=dt,
                has_media=has_media,
                media_type=media_type,
                share_link=share_link,
                share_text=share_text,
                reactions=reactions,
            )
        )

    return participants, messages


def load_thread(thread_dir: str | Path) -> tuple[list[str], list[Message]]:
    """Load and merge all message files for a single Instagram conversation.

    A conversation directory contains one or more message_N.json files. This
    function reads them all, concatenates the messages, and returns them sorted
    chronologically by timestamp.

    Args:
        thread_dir: Path to the conversation folder (e.g. inbox/username_abc123/).

    Returns:
        Tuple of (participant name list, time-sorted Message objects).
    """
    thread_dir = Path(thread_dir)
    thread_id = thread_dir.name
    all_participants: list[str] = []
    all_messages: list[Message] = []

    for msg_file in sorted(thread_dir.glob("message_*.json")):
        participants, messages = _parse_file(msg_file, thread_id)
        # Participant list is consistent across files; take it from the first one.
        if not all_participants:
            all_participants = participants
        all_messages.extend(messages)

    # Files may be ordered newest-first (Instagram's default), so sort explicitly.
    all_messages.sort(key=lambda m: m.timestamp)
    return all_participants, all_messages


def load_all_threads(
    inbox_path: str | Path,
) -> dict[str, tuple[list[str], list[Message]]]:
    """Load every conversation thread found under an Instagram inbox directory.

    Iterates over each subdirectory of *inbox_path* and calls `load_thread` on
    it. Directories that yield no messages (e.g. empty or non-conversation
    folders) are excluded from the result.

    Args:
        inbox_path: Path to the top-level inbox/ folder from an Instagram export.

    Returns:
        Dict mapping thread_id (directory name) to (participants, messages) tuples.
    """
    inbox = Path(inbox_path)
    threads: dict[str, tuple[list[str], list[Message]]] = {}

    for thread_dir in sorted(inbox.iterdir()):
        if not thread_dir.is_dir():
            continue
        participants, messages = load_thread(thread_dir)
        # Skip folders that produced no parseable messages.
        if messages:
            threads[thread_dir.name] = (participants, messages)

    return threads
