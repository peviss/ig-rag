import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { SourceCard } from "./SourceCard";
import type { ChatMessage } from "../types";

interface Props {
  message: ChatMessage;
}

const INTENT_LABELS: Record<string, string> = {
  search: "search",
  summarize: "summarize",
  timeline: "timeline",
  count: "count",
  comparison: "compare",
};

export function MessageBubble({ message }: Props) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="message user-message">
        <div className="bubble user-bubble">{message.text}</div>
      </div>
    );
  }

  if (message.loading) {
    return (
      <div className="message ai-message">
        <div className="bubble ai-bubble loading-bubble">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
      </div>
    );
  }

  const sources = message.sources ?? [];
  const intent = message.intent;

  return (
    <div className="message ai-message">
      <div className="bubble ai-bubble">
        {message.has_llm_answer ? (
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
          </div>
        ) : (
          <p className="no-llm-note">
            No API key — showing {sources.length} retrieved excerpts below.
          </p>
        )}

        {intent && (
          <div className="intent-footer">
            {intent.type !== "search" && (
              <span className="intent-tag">{INTENT_LABELS[intent.type] ?? intent.type}</span>
            )}
            {intent.temporal_anchor && (
              <span className="intent-tag">anchor: {intent.temporal_anchor}</span>
            )}
            {intent.entities.slice(0, 3).map((e) => (
              <span key={e} className="entity-tag">{e}</span>
            ))}
          </div>
        )}

        {sources.length > 0 && (
          <button
            className="sources-toggle"
            onClick={() => setShowSources((s) => !s)}
          >
            {showSources ? "Hide" : "Show"} {sources.length} source
            {sources.length !== 1 ? "s" : ""}
          </button>
        )}
      </div>

      {showSources && (
        <div className="sources-list">
          {sources.map((s) => (
            <SourceCard key={s.chunk_id} source={s} index={s.rank} />
          ))}
        </div>
      )}
    </div>
  );
}
