"""FastAPI backend for the Instagram RAG pipeline."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

STORE_DIR = Path("./ig_store")
MSG_STORE_PATH = STORE_DIR / "messages.pkl"
BM25_PATH = STORE_DIR / "bm25.pkl"
CHROMA_DIR = STORE_DIR / "chroma"

_engine = None
_msg_store = None


def _load_engine(model: str = "paraphrase-multilingual-MiniLM-L12-v2") -> object:
    from src.bm25_index import BM25Index
    from src.chroma_store import ChromaStore
    from src.embeddings import MultilingualEmbeddings
    from src.message_store import MessageStore
    from src.query_engine import QueryEngine
    from src.retriever import HybridRetriever

    msg_store = MessageStore.from_file(MSG_STORE_PATH)
    bm25 = BM25Index.load(BM25_PATH)
    embeddings = MultilingualEmbeddings(model_name=model)
    chroma = ChromaStore(CHROMA_DIR, embeddings)
    retriever = HybridRetriever(chroma, bm25, msg_store)
    return QueryEngine(retriever)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _msg_store
    if MSG_STORE_PATH.exists() and BM25_PATH.exists():
        print("Loading RAG engine…")
        from src.message_store import MessageStore
        _msg_store = MessageStore.from_file(MSG_STORE_PATH)
        _engine = _load_engine()
        print("Engine ready.")
    else:
        print("WARNING: index not found. Run `python main.py ingest` first.")
    yield


app = FastAPI(title="Instagram RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_engine():
    if _engine is None:
        raise HTTPException(503, "Index not loaded. Run `python main.py ingest` first.")
    return _engine


# ── Request models ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    top_k: int = 8


class SearchRequest(BaseModel):
    query: str
    k: int = 10


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/api/ask")
async def ask(req: AskRequest):
    engine = _require_engine()
    result = engine.run(req.query, top_k=req.top_k)
    return {
        "answer": result.answer,
        "has_llm_answer": result.has_llm_answer,
        "query": result.query,
        "intent": result.intent.model_dump() if result.intent else None,
        "sources": [
            {
                "rank": r.rank,
                "score": round(r.score, 4),
                "chunk_id": r.chunk_id,
                "chunk": r.chunk.model_dump(),
            }
            for r in result.sources
        ],
    }


@app.post("/api/search")
async def search(req: SearchRequest):
    engine = _require_engine()
    intent = engine.understand(req.query)
    candidates = engine.retrieve(req.query, intent, k_candidates=req.k)
    top = engine._rerank(req.query, candidates, top_k=req.k)
    return {
        "query": req.query,
        "intent": intent.model_dump(),
        "results": [
            {
                "rank": i,
                "score": round(score, 4),
                "chunk_id": chunk.chunk_id,
                "chunk": chunk.model_dump(),
            }
            for i, (chunk, score) in enumerate(top)
        ],
    }


@app.get("/api/stats")
async def stats():
    if not MSG_STORE_PATH.exists():
        raise HTTPException(503, "Index not found. Run `python main.py ingest` first.")

    from src.bm25_index import BM25Index
    from src.message_store import MessageStore

    msg_store = MessageStore.from_file(MSG_STORE_PATH)
    bm25 = BM25Index.load(BM25_PATH)
    msgs = msg_store.messages

    sender_counts: dict[str, int] = {}
    for m in msgs:
        sender_counts[m.sender] = sender_counts.get(m.sender, 0) + 1

    date_range = None
    if msgs:
        start = datetime.fromtimestamp(msgs[0].timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        end = datetime.fromtimestamp(msgs[-1].timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        date_range = {"start": start, "end": end}

    chroma_counts: dict[str, int] = {}
    if CHROMA_DIR.exists():
        from src.chroma_store import ChromaStore
        from src.embeddings import MultilingualEmbeddings

        emb = MultilingualEmbeddings()
        chroma = ChromaStore(CHROMA_DIR, emb)
        chroma_counts = {"l2": chroma.count(2), "l3": chroma.count(3)}

    return {
        "message_count": len(msgs),
        "bm25_size": bm25.size,
        "date_range": date_range,
        "senders": sender_counts,
        "chroma": chroma_counts,
        "index_ready": True,
    }


@app.get("/api/stats/messages")
async def stats_messages(
    start: str | None = None,
    end: str | None = None,
    tz_offset: int = -3,
):
    from collections import defaultdict
    from datetime import timedelta

    if _msg_store is None:
        raise HTTPException(503, "Index not loaded. Run `python main.py ingest` first.")

    msgs = _msg_store.messages

    # Date filtering
    start_ts = end_ts = None
    if start:
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    if end:
        # inclusive: add one full day
        end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 86400

    if start_ts or end_ts:
        msgs = [
            m for m in msgs
            if (start_ts is None or m.timestamp >= start_ts)
            and (end_ts is None or m.timestamp < end_ts)
        ]

    if not msgs:
        return {"total": 0, "senders": [], "by_sender": {}, "by_hour": [],
                "by_weekday": [], "by_month": [], "top_days": [], "summary": {}}

    senders = sorted({m.sender for m in msgs})

    # ── Per-sender ─────────────────────────────────────────────────────────────
    counts: dict[str, int] = defaultdict(int)
    lengths: dict[str, list[int]] = defaultdict(list)
    media_counts: dict[str, int] = defaultdict(int)

    for m in msgs:
        counts[m.sender] += 1
        if m.text:
            lengths[m.sender].append(len(m.text))
        if m.has_media:
            media_counts[m.sender] += 1

    total = len(msgs)
    by_sender = {
        s: {
            "count": counts[s],
            "avg_length": round(sum(lengths[s]) / len(lengths[s])) if lengths[s] else 0,
            "media_count": media_counts[s],
            "pct": round(counts[s] / total * 100, 1),
        }
        for s in senders
    }

    # ── By hour & weekday (local timezone) ────────────────────────────────────
    local_tz = timezone(timedelta(hours=tz_offset))
    hour_buckets: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in msgs:
        h = datetime.fromtimestamp(m.timestamp, tz=local_tz).hour
        hour_buckets[h][m.sender] += 1

    by_hour = [
        {"hour": h, **{s: hour_buckets[h].get(s, 0) for s in senders}}
        for h in range(24)
    ]

    # ── By weekday ─────────────────────────────────────────────────────────────
    DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_buckets: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in msgs:
        wd = datetime.fromtimestamp(m.timestamp, tz=local_tz).weekday()
        weekday_buckets[wd][m.sender] += 1

    by_weekday = [
        {"day": DAYS[d], **{s: weekday_buckets[d].get(s, 0) for s in senders}}
        for d in range(7)
    ]

    # ── By month ───────────────────────────────────────────────────────────────
    month_buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in msgs:
        dt = datetime.fromtimestamp(m.timestamp, tz=timezone.utc)
        key = f"{dt.year}-{dt.month:02d}"
        month_buckets[key][m.sender] += 1

    by_month = [
        {"month": k, **{s: month_buckets[k].get(s, 0) for s in senders}}
        for k in sorted(month_buckets)
    ]

    # ── Top active days ────────────────────────────────────────────────────────
    day_counts: dict[str, int] = defaultdict(int)
    day_sender: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in msgs:
        day = datetime.fromtimestamp(m.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        day_counts[day] += 1
        day_sender[day][m.sender] += 1

    top_days = [
        {"date": d, "count": c, **{s: day_sender[d].get(s, 0) for s in senders}}
        for d, c in sorted(day_counts.items(), key=lambda x: -x[1])[:10]
    ]

    # ── Summary ────────────────────────────────────────────────────────────────
    first_dt = datetime.fromtimestamp(msgs[0].timestamp, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(msgs[-1].timestamp, tz=timezone.utc)
    active_days_set = {
        datetime.fromtimestamp(m.timestamp, tz=timezone.utc).date() for m in msgs
    }
    total_days = (last_dt.date() - first_dt.date()).days + 1
    avg_per_day = round(total / max(len(active_days_set), 1), 1)

    # Longest streak (consecutive active days)
    sorted_days = sorted(active_days_set)
    streak = best_streak = 1
    for i in range(1, len(sorted_days)):
        if (sorted_days[i] - sorted_days[i - 1]).days == 1:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 1

    # Longest gap (days with no messages)
    longest_gap = 0
    for i in range(1, len(sorted_days)):
        gap = (sorted_days[i] - sorted_days[i - 1]).days - 1
        longest_gap = max(longest_gap, gap)

    summary = {
        "first_date": first_dt.strftime("%Y-%m-%d"),
        "last_date": last_dt.strftime("%Y-%m-%d"),
        "active_days": len(active_days_set),
        "total_days": total_days,
        "avg_per_day": avg_per_day,
        "longest_streak": best_streak,
        "longest_gap": longest_gap,
    }

    return {
        "total": total,
        "senders": senders,
        "by_sender": by_sender,
        "by_hour": by_hour,
        "by_weekday": by_weekday,
        "by_month": by_month,
        "top_days": top_days,
        "summary": summary,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "index_loaded": _engine is not None}
