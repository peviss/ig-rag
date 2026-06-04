import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { MessageBubble } from "./MessageBubble";
import type { ChatMessage, ChatMode } from "../types";

interface Props {
  mode: ChatMode;
  topK: number;
}

let idCounter = 0;
const uid = () => String(++idCounter);

export function ChatView({ mode, topK }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function submit() {
    const q = input.trim();
    if (!q || busy) return;

    const userMsg: ChatMessage = { id: uid(), role: "user", text: q };
    const loadingId = uid();
    const loadingMsg: ChatMessage = {
      id: loadingId,
      role: "assistant",
      text: "",
      loading: true,
    };

    setMessages((prev) => [...prev, userMsg, loadingMsg]);
    setInput("");
    setBusy(true);

    try {
      let aiMsg: ChatMessage;
      if (mode === "ask") {
        const res = await api.ask(q, topK);
        aiMsg = {
          id: loadingId,
          role: "assistant",
          text: res.answer,
          sources: res.sources,
          intent: res.intent,
          has_llm_answer: res.has_llm_answer,
          loading: false,
        };
      } else {
        const res = await api.search(q, topK);
        aiMsg = {
          id: loadingId,
          role: "assistant",
          text: "",
          sources: res.results,
          intent: res.intent,
          has_llm_answer: false,
          loading: false,
        };
      }
      setMessages((prev) => prev.map((m) => (m.id === loadingId ? aiMsg : m)));
    } catch (err: unknown) {
      const errMsg: ChatMessage = {
        id: loadingId,
        role: "assistant",
        text: `Error: ${err instanceof Error ? err.message : String(err)}`,
        loading: false,
        has_llm_answer: false,
      };
      setMessages((prev) => prev.map((m) => (m.id === loadingId ? errMsg : m)));
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="chat-view">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <p className="empty-icon">{mode === "ask" ? "🤖" : "🔍"}</p>
            <p className="empty-title">
              {mode === "ask"
                ? "Ask a question — Claude will answer"
                : "Search for messages — no AI, just results"}
            </p>
            <p className="empty-subtitle">
              {mode === "ask"
                ? "Retrieves relevant chunks, then Claude synthesizes a cited answer."
                : "Retrieves the most relevant message chunks and ranks them by relevance."}
            </p>
          </div>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="input-bar">
        <textarea
          ref={inputRef}
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={mode === "ask" ? "Ask a question… (Enter to send)" : "Search messages… (Enter to search)"}
          rows={1}
          disabled={busy}
        />
        <button className="send-btn" onClick={submit} disabled={busy || !input.trim()}>
          {busy ? (
            <span className="spinner" />
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M22 2L11 13M22 2L15 22L11 13L2 9L22 2z" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
