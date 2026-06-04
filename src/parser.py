from __future__ import annotations
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from .models import Message


def fix_encoding(text: str) -> str:
    """Fix Instagram's mojibake: bytes stored as Latin-1 chars re-encoded as UTF-8."""
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _msg_id(thread_id: str, ts_ms: int, sender: str, idx: int) -> str:
    key = f"{thread_id}:{ts_ms}:{sender}:{idx}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _parse_file(path: Path, thread_id: str) -> tuple[list[str], list[Message]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    participants = [fix_encoding(p["name"]) for p in data.get("participants", [])]
    messages: list[Message] = []

    for idx, raw in enumerate(data.get("messages", [])):
        sender = fix_encoding(raw.get("sender_name", "unknown"))
        ts_ms = raw.get("timestamp_ms", 0)
        ts = ts_ms // 1000
        content = fix_encoding(raw.get("content", "") or "")

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
            if not content:
                content = f"[Compartido] {share_text}" if share_text else "[Enlace compartido]"

        reactions = [
            {
                "actor": fix_encoding(r.get("actor", "")),
                "reaction": fix_encoding(r.get("reaction", "")),
            }
            for r in raw.get("reactions", [])
        ]

        if not content and not has_media:
            continue
        if not content:
            content = f"[{media_type}]"

        # Replace auto-generated attachment placeholders with the real share text
        _attachment_phrases = (
            "adjunto",      # covers all "enviado un archivo adjunto" variants
            "attachment",   # English export variant
        )
        if any(phrase in content.lower() for phrase in _attachment_phrases):
            if share_text:
                content = f"[Compartido] {share_text}"
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
    thread_dir = Path(thread_dir)
    thread_id = thread_dir.name
    all_participants: list[str] = []
    all_messages: list[Message] = []

    for msg_file in sorted(thread_dir.glob("message_*.json")):
        participants, messages = _parse_file(msg_file, thread_id)
        if not all_participants:
            all_participants = participants
        all_messages.extend(messages)

    all_messages.sort(key=lambda m: m.timestamp)
    return all_participants, all_messages


def load_all_threads(
    inbox_path: str | Path,
) -> dict[str, tuple[list[str], list[Message]]]:
    inbox = Path(inbox_path)
    threads: dict[str, tuple[list[str], list[Message]]] = {}

    for thread_dir in sorted(inbox.iterdir()):
        if not thread_dir.is_dir():
            continue
        participants, messages = load_thread(thread_dir)
        if messages:
            threads[thread_dir.name] = (participants, messages)

    return threads
