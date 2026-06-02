import json
import os
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ── 1. Parse Instagram export ──────────────────────────────────────────────────

def load_instagram_thread(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    participants = [p["name"] for p in data.get("participants", [])]
    messages = []
    for m in data.get("messages", []):
        content = m.get("content", "").strip()
        if not content:
            continue  # skip photos/reactions with no text
        messages.append({
            "sender": m.get("sender_name", "unknown"),
            "text": content,
            "ts": m.get("timestamp_ms", 0) // 1000,  # → unix seconds
        })

    # Instagram exports newest-first — reverse to chronological order
    messages.reverse()
    return participants, messages


# ── 2. Conversation-window chunking ───────────────────────────────────────────

def chunk_conversation(
    participants: list[str],
    messages: list[dict],
    thread_name: str,
    window: int = 20,   # messages per chunk
    overlap: int = 5,   # overlap between chunks
) -> list[Document]:
    """
    Groups messages into sliding windows so the LLM sees full conversational
    context rather than isolated one-liners.
    """
    docs = []
    step = window - overlap
    for i in range(0, len(messages), step):
        batch = messages[i : i + window]
        if not batch:
            break

        # Format as readable transcript
        lines = []
        for m in batch:
            date = datetime.utcfromtimestamp(m["ts"]).strftime("%Y-%m-%d %H:%M")
            lines.append(f"[{date}] {m['sender']}: {m['text']}")
        text = "\n".join(lines)

        # Metadata enables filtered search later
        docs.append(Document(
            page_content=text,
            metadata={
                "thread": thread_name,
                "participants": ", ".join(participants),
                "start_ts": batch[0]["ts"],
                "end_ts": batch[-1]["ts"],
                "start_date": datetime.utcfromtimestamp(batch[0]["ts"]).strftime("%Y-%m-%d"),
                "msg_count": len(batch),
            },
        ))
    return docs


# ── 3. Load all threads from export folder ────────────────────────────────────

def load_all_threads(inbox_path: str) -> list[Document]:
    """
    Expects the Instagram export structure:
      messages/inbox/<thread_name>/message_1.json
    """
    all_docs = []
    inbox = Path(inbox_path)
    for thread_dir in inbox.iterdir():
        if not thread_dir.is_dir():
            continue
        for msg_file in sorted(thread_dir.glob("message_*.json")):
            participants, messages = load_instagram_thread(str(msg_file))
            if not messages:
                continue
            docs = chunk_conversation(
                participants=participants,
                messages=messages,
                thread_name=thread_dir.name,
            )
            all_docs.extend(docs)
            print(f"  {thread_dir.name}: {len(messages)} messages → {len(docs)} chunks")
    return all_docs


# ── 4. Embed + index ──────────────────────────────────────────────────────────

class SentenceTransformerEmbeddings(Embeddings):
    """Local embeddings via sentence-transformers — no API key required."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


def build_vectorstore(
    docs: list[Document],
    persist_dir: str = "./instagram_db",
    embedding_provider: str = "openai",  # "openai" | "sentence-transformers"
    st_model: str = "all-MiniLM-L6-v2",
) -> Chroma:
    if embedding_provider == "sentence-transformers":
        embeddings: Embeddings = SentenceTransformerEmbeddings(st_model)
        print(f"Using local sentence-transformers model: {st_model}")
    else:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.environ["OPENAI_API_KEY"],
        )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="instagram_chats",
    )
    print(f"\nStored {vectorstore._collection.count()} chunks in {persist_dir}")
    return vectorstore


# ── 5. Query examples ─────────────────────────────────────────────────────────

def search(vectorstore: Chroma, query: str, person: str = None, k: int = 4):
    """
    Semantic search with optional filter by participant name.
    """
    where = {"participants": {"$contains": person}} if person else None
    results = vectorstore.similarity_search(query, k=k, filter=where)
    for r in results:
        print(f"\n── {r.metadata['thread']} | {r.metadata['start_date']} ──")
        print(r.page_content[:400])


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Instagram export…")
    # Point this to: <your_export>/messages/inbox/
    docs = load_all_threads("./messages/inbox")
    print(f"\nTotal chunks: {len(docs)}")

    print("\nBuilding vectorstore…")
    # Switch to "sentence-transformers" to run fully offline without an API key.
    vs = build_vectorstore(docs, embedding_provider="sentence-transformers")

    # Example queries
    print("\n── Query: plans for the weekend ──")
    search(vs, "plans for the weekend")

    print("\n── Query: restaurant recommendations (filtered to one person) ──")
    search(vs, "restaurant recommendations", person="Maria")

    # Reload later without re-embedding (must use the same embedding as at index time):
    # vs = Chroma(persist_directory="./instagram_db",
    #             embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    #             collection_name="instagram_chats")
    # vs = Chroma(persist_directory="./instagram_db",
    #             embedding_function=SentenceTransformerEmbeddings("all-MiniLM-L6-v2"),
    #             collection_name="instagram_chats")