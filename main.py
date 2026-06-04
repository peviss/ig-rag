"""
Instagram RAG — entry point.

Commands:
  ingest   Parse export, build BM25 index, embed & store L2/L3 chunks.
  ask      Answer a natural language question about the conversation.
  search   Raw retrieval (no LLM answer).
  chat     Interactive Q&A session.
  stats    Show index statistics.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich.text import Text
from rich.rule import Rule

load_dotenv()
console = Console()

# ── Store paths ────────────────────────────────────────────────────────────────
STORE_DIR = Path("./ig_store")
MSG_STORE_PATH = STORE_DIR / "messages.pkl"
BM25_PATH = STORE_DIR / "bm25.pkl"
CHROMA_DIR = STORE_DIR / "chroma"


# ── Ingest ─────────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    from src.parser import load_all_threads
    from src.chunker import chunk_all
    from src.embeddings import MultilingualEmbeddings
    from src.message_store import MessageStore
    from src.chroma_store import ChromaStore
    from src.bm25_index import BM25Index

    inbox = Path(args.inbox)
    if not inbox.exists():
        console.print(f"[red]Inbox not found: {inbox}[/red]")
        sys.exit(1)

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]1 / 4 — Parsing Instagram export[/bold]")
    threads = load_all_threads(inbox)
    all_messages = []
    all_l2: list = []
    all_l3: list = []

    for thread_id, (participants, messages) in threads.items():
        console.print(
            f"  [cyan]{thread_id}[/cyan]: {len(messages):,} messages  "
            f"participants: {', '.join(participants)}"
        )
        all_messages.extend(messages)
        l2, l3 = chunk_all(messages, thread_id, participants)
        all_l2.extend(l2)
        all_l3.extend(l3)

    console.print(
        f"\n[green]Total:[/green] {len(all_messages):,} messages  "
        f"| {len(all_l2):,} L2 chunks  | {len(all_l3):,} L3 segments"
    )

    # ── Message store ──────────────────────────────────────────────────────────
    console.rule("[bold]2 / 4 — Saving message store[/bold]")
    msg_store = MessageStore()
    msg_store.load(all_messages)
    msg_store.save(MSG_STORE_PATH)
    console.print(f"  Saved {len(msg_store):,} messages → {MSG_STORE_PATH}")

    # ── BM25 ───────────────────────────────────────────────────────────────────
    console.rule("[bold]3 / 4 — Building BM25 index[/bold]")
    bm25 = BM25Index()
    items = [(m.id, m.text) for m in all_messages]
    bm25.build(items)
    bm25.save(BM25_PATH)
    console.print(f"  BM25 index: {bm25.size:,} documents → {BM25_PATH}")

    # ── Dense embeddings ───────────────────────────────────────────────────────
    console.rule("[bold]4 / 4 — Embedding L2 + L3 chunks[/bold]")
    embeddings = MultilingualEmbeddings(model_name=args.model, batch_size=args.batch_size)
    console.print(f"  Model: [cyan]{args.model}[/cyan]  dim={embeddings.dimension}")

    chroma = ChromaStore(CHROMA_DIR, embeddings)
    if chroma.is_populated() and not args.force:
        console.print(
            "[yellow]Chroma already populated. Use --force to re-embed.[/yellow]"
        )
    else:
        console.print(f"  Embedding {len(all_l2):,} L2 chunks…")
        chroma.add_chunks(all_l2)
        console.print(f"  Embedding {len(all_l3):,} L3 segments…")
        chroma.add_chunks(all_l3)
        console.print(
            f"  [green]Stored:[/green] {chroma.count(2):,} L2  |  {chroma.count(3):,} L3"
        )

    console.rule("[bold green]Ingestion complete[/bold green]")


# ── Load components ────────────────────────────────────────────────────────────

def _load_components(model: str, rerank: bool):
    from src.embeddings import MultilingualEmbeddings
    from src.message_store import MessageStore
    from src.chroma_store import ChromaStore
    from src.bm25_index import BM25Index
    from src.retriever import HybridRetriever
    from src.reranker import CrossEncoderReranker
    from src.query_engine import QueryEngine

    if not MSG_STORE_PATH.exists() or not BM25_PATH.exists():
        console.print(
            "[red]Index not found. Run [bold]python main.py ingest[/bold] first.[/red]"
        )
        sys.exit(1)

    with console.status("Loading message store…"):
        msg_store = MessageStore.from_file(MSG_STORE_PATH)

    with console.status("Loading BM25 index…"):
        bm25 = BM25Index.load(BM25_PATH)

    with console.status(f"Loading embedding model [cyan]{model}[/cyan]…"):
        embeddings = MultilingualEmbeddings(model_name=model)

    with console.status("Connecting to Chroma…"):
        chroma = ChromaStore(CHROMA_DIR, embeddings)

    retriever = HybridRetriever(chroma, bm25, msg_store)

    reranker = None
    if rerank:
        with console.status("Loading cross-encoder reranker…"):
            reranker = CrossEncoderReranker()

    engine = QueryEngine(retriever, reranker=reranker)
    return engine


# ── Ask ────────────────────────────────────────────────────────────────────────

def cmd_ask(args: argparse.Namespace) -> None:
    engine = _load_components(args.model, args.rerank)
    _run_query(engine, args.query, show_sources=args.sources, top_k=args.top_k)


_SENDER_STYLES = ["bold blue", "bold magenta", "bold green", "bold yellow"]


def _colorize_transcript(text: str, max_lines: int | None = None) -> Text:
    """Render a conversation transcript with colored timestamps and sender names."""
    import re

    sender_colors: dict[str, str] = {}
    rich_text = Text()
    lines = text.split("\n")
    if max_lines:
        lines = lines[:max_lines]

    for line in lines:
        m = re.match(r"(\[[\d\-: ]+\]) ([^:]+): (.*)", line)
        if m:
            ts, sender, msg = m.groups()
            if sender not in sender_colors:
                sender_colors[sender] = _SENDER_STYLES[len(sender_colors) % len(_SENDER_STYLES)]
            rich_text.append(ts + " ", style="dim")
            rich_text.append(sender + ": ", style=sender_colors[sender])
            rich_text.append(msg + "\n")
        else:
            rich_text.append(line + "\n", style="dim italic")

    return rich_text


def _render_source_panel(r, idx: int) -> Panel:
    """Render one retrieval result as a styled panel."""
    chunk = r.chunk
    date_str = chunk.start_date if chunk.start_date == chunk.end_date else f"{chunk.start_date} → {chunk.end_date}"
    title = (
        f"[dim]#{idx}[/dim]  "
        f"[cyan]{date_str}[/cyan]  "
        f"[dim]{chunk.message_count} msgs · L{chunk.level} · score {r.score:.3f}[/dim]"
    )
    transcript = _colorize_transcript(chunk.text)
    return Panel(transcript, title=title, title_align="left", border_style="dim", padding=(0, 1))


def _render_intent_footer(intent) -> None:
    if not intent:
        return
    parts = []
    if intent.type != "search":
        parts.append(f"[yellow]{intent.type}[/yellow]")
    if intent.temporal_anchor:
        parts.append(f"anchor:[yellow]{intent.temporal_anchor}[/yellow]")
    if intent.entities:
        parts.append("entities:" + ", ".join(f"[italic]{e}[/italic]" for e in intent.entities[:4]))
    if parts:
        console.print("  [dim]" + "  ·  ".join(parts) + "[/dim]")


def _run_query(engine, query: str, show_sources: bool = False, top_k: int = 8) -> None:
    with console.status("[bold]Thinking…[/bold]"):
        result = engine.run(query, top_k=top_k)

    console.print()

    if result.has_llm_answer:
        console.print(Panel(
            Markdown(result.answer),
            title=f"[bold cyan]{query}[/bold cyan]",
            title_align="left",
            border_style="cyan",
            padding=(1, 2),
        ))
        _render_intent_footer(result.intent)

        if show_sources and result.sources:
            console.print()
            console.print(Rule("[dim]Sources[/dim]", style="dim"))
            for r in result.sources:
                console.print(_render_source_panel(r, r.rank + 1))
    else:
        # No LLM — show retrieved excerpts directly
        console.print(f"[dim]No ANTHROPIC_API_KEY — showing {len(result.sources)} retrieved excerpts[/dim]\n")
        for r in result.sources:
            console.print(_render_source_panel(r, r.rank + 1))
            console.print()
        _render_intent_footer(result.intent)


# ── Search ─────────────────────────────────────────────────────────────────────

def cmd_search(args: argparse.Namespace) -> None:
    engine = _load_components(args.model, args.rerank)

    with console.status("Retrieving…"):
        intent = engine.understand(args.query)
        candidates = engine.retrieve(args.query, intent, k_candidates=args.k)
        top = engine._rerank(args.query, candidates, top_k=args.k)

    console.print(f"\n[bold]Top {len(top)} results for:[/bold] [cyan]{args.query}[/cyan]\n")
    from src.models import RetrievalResult
    for i, (chunk, score) in enumerate(top, 1):
        r = RetrievalResult(chunk_id=chunk.chunk_id, chunk=chunk, score=score, rank=i - 1)
        console.print(_render_source_panel(r, i))
        console.print()


# ── Stats ──────────────────────────────────────────────────────────────────────

def cmd_stats(_args: argparse.Namespace) -> None:
    from src.message_store import MessageStore
    from src.bm25_index import BM25Index

    if not MSG_STORE_PATH.exists():
        console.print("[red]Index not found. Run ingest first.[/red]")
        return

    msg_store = MessageStore.from_file(MSG_STORE_PATH)
    bm25 = BM25Index.load(BM25_PATH)

    msgs = msg_store.messages
    t = Table(title="Index Statistics")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")
    t.add_row("Total messages", f"{len(msgs):,}")
    t.add_row("BM25 index size", f"{bm25.size:,}")

    if msgs:
        from datetime import datetime, timezone
        start = datetime.fromtimestamp(msgs[0].timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        end = datetime.fromtimestamp(msgs[-1].timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        t.add_row("Date range", f"{start} → {end}")

        senders = {}
        for m in msgs:
            senders[m.sender] = senders.get(m.sender, 0) + 1
        for sender, count in sorted(senders.items(), key=lambda x: -x[1]):
            t.add_row(f"  {sender}", f"{count:,} messages ({count/len(msgs)*100:.1f}%)")

    if CHROMA_DIR.exists():
        from src.embeddings import MultilingualEmbeddings
        from src.chroma_store import ChromaStore
        emb = MultilingualEmbeddings()
        chroma = ChromaStore(CHROMA_DIR, emb)
        t.add_row("Chroma L2 chunks", f"{chroma.count(2):,}")
        t.add_row("Chroma L3 segments", f"{chroma.count(3):,}")

    console.print(t)


# ── Chat ───────────────────────────────────────────────────────────────────────

def cmd_chat(args: argparse.Namespace) -> None:
    engine = _load_components(args.model, args.rerank)
    console.print(
        Panel(
            "[bold]Instagram RAG Chat[/bold]\n"
            "Type your question and press Enter. Type [cyan]exit[/cyan] to quit.",
            border_style="cyan",
        )
    )
    while True:
        try:
            query = console.input("\n[bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue
        _run_query(engine, query, show_sources=args.sources, top_k=args.top_k)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ig-rag",
        description="Instagram DM RAG pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Parse + embed + index the Instagram export")
    p_ingest.add_argument(
        "--inbox",
        default="./messages/inbox",
        help="Path to messages/inbox directory (default: ./messages/inbox)",
    )
    p_ingest.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model for embeddings",
    )
    p_ingest.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p_ingest.add_argument(
        "--force", action="store_true", help="Re-embed even if Chroma is populated"
    )

    # ask
    p_ask = sub.add_parser("ask", help="Answer a question about the conversation")
    p_ask.add_argument("query", help="Natural language question")
    p_ask.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    p_ask.add_argument("--top-k", type=int, default=8, dest="top_k")
    p_ask.add_argument("--sources", action="store_true", help="Show source chunks")
    p_ask.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranker")

    # search
    p_search = sub.add_parser("search", help="Raw hybrid retrieval (no LLM answer)")
    p_search.add_argument("query")
    p_search.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    p_search.add_argument("-k", type=int, default=10)
    p_search.add_argument("--rerank", action="store_true")

    # chat
    p_chat = sub.add_parser("chat", help="Interactive Q&A session")
    p_chat.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    p_chat.add_argument("--top-k", type=int, default=8, dest="top_k")
    p_chat.add_argument("--sources", action="store_true")
    p_chat.add_argument("--rerank", action="store_true")

    # stats
    p_stats = sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()
    dispatch = {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "search": cmd_search,
        "chat": cmd_chat,
        "stats": cmd_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
