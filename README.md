# ig-rag

RAG pipeline for Instagram DM exports. Ask natural language questions about your conversation history and get answers grounded strictly in the exported messages.

```
python main.py ask "When did we talk about moving?"
python main.py ask "Conversations about restaurant recomendations" --sources
python main.py chat
```

## Setup

**Docker (recommended):**

```bash
# 1. Add your API key to .env
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 2. Build the image (bakes in the embedding model, ~3–4 min first time)
make build

# 3. Ingest your export
make ingest
```

**Local:**

```bash
uv sync
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python main.py ingest
```

Without `ANTHROPIC_API_KEY` the system runs in retrieval-only mode (returns raw matching excerpts).

## Ingest

Point at your Instagram export's `messages/inbox` directory:

```bash
make ingest
# or
python main.py ingest --inbox ./messages/inbox
```

This builds three indexes in `./ig_store/`:
- BM25 index over all individual messages (keyword recall)
- ChromaDB L2 collection — sliding-window conversation chunks (semantic recall)
- ChromaDB L3 collection — topic segments grouped by time gaps

Default embedding model: `paraphrase-multilingual-MiniLM-L12-v2` (fast, multilingual).
For better Spanish retrieval upgrade with `--model intfloat/multilingual-e5-large`.

## Commands

| Make target | Equivalent | Description |
|---|---|---|
| `make ingest` | `python main.py ingest` | Parse export, build BM25 + Chroma indexes |
| `make ask QUERY="..."` | `python main.py ask "..."` | Answer a question with LLM + citations |
| `make search QUERY="..."` | `python main.py search "..."` | Raw hybrid retrieval, no LLM |
| `make chat` | `python main.py chat` | Interactive Q&A session |
| `make stats` | `python main.py stats` | Show index statistics |
| `make clean` | `rm -rf ig_store/` | Wipe the index |

### Options (local / `docker compose run`)

```bash
python main.py ask "..." --sources        # show retrieved source chunks
python main.py ask "..." --rerank         # enable cross-encoder reranker
python main.py ask "..." --top-k 12       # number of chunks passed to LLM
python main.py search "..." -k 20         # number of retrieval results
python main.py ingest --force             # re-embed even if index exists
```

## Architecture

```
src/
  parser.py          Instagram JSON parser — fixes mojibake encoding
  chunker.py         L2 sliding windows + L3 temporal topic segments
  embeddings.py      Multilingual sentence-transformers wrapper (E5-aware)
  message_store.py   In-memory message store with context-window expansion
  chroma_store.py    ChromaDB multi-collection (L2 + L3)
  bm25_index.py      BM25Okapi index over all messages
  retriever.py       Hybrid RRF retrieval (dense + BM25)
  reranker.py        Cross-encoder reranker (multilingual mMiniLM)
  query_engine.py    Intent detection → query expansion → retrieve → answer
  models.py          Pydantic models
```

**Retrieval pipeline:**
1. Query understanding — intent, entities, temporal anchor, expanded queries (Claude if API key present, rule-based otherwise)
2. Hybrid retrieval — dense search on L2/L3 Chroma + BM25 on L1 messages expanded to context windows
3. RRF fusion — 60% dense, 30% BM25
4. Temporal re-sort — for *first/last/earliest/latest* queries, re-order by timestamp before LLM
5. Cross-encoder reranker — optional, `--rerank` flag
6. LLM answer — Claude Sonnet with strict grounding rules and timestamp citations

## Data format

Expects the standard Instagram export structure:

```
messages/
  inbox/
    <thread_name>/
      message_1.json
      message_2.json
      ...
```

Supports: text, shared links, photos, videos, audio, reactions.
