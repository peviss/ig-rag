import { useState } from "react";
import type { Source } from "../types";

interface Props {
  source: Source;
  index: number;
}

const SENDER_COLORS = ["#818cf8", "#f472b6", "#34d399", "#fbbf24", "#60a5fa"];

function colorizeTranscript(text: string, colors: Record<string, string>): React.ReactNode[] {
  const lines = text.split("\n");
  return lines.map((line, i) => {
    const m = line.match(/^(\[[\d\-: ]+\]) ([^:]+): (.*)/);
    if (m) {
      const [, ts, sender, msg] = m;
      const color = colors[sender] ?? "#a0a0b8";
      return (
        <span key={i} style={{ display: "block" }}>
          <span style={{ color: "#4a4a6a", fontSize: "0.75em" }}>{ts} </span>
          <span style={{ color, fontWeight: 600 }}>{sender}: </span>
          <span>{msg}</span>
          {"\n"}
        </span>
      );
    }
    return (
      <span key={i} style={{ display: "block", color: "#5a5a7a", fontStyle: "italic" }}>
        {line || " "}
      </span>
    );
  });
}

export function SourceCard({ source }: Props) {
  const [open, setOpen] = useState(false);
  const { chunk, score, rank } = source;

  const dateLabel =
    chunk.start_date === chunk.end_date
      ? chunk.start_date
      : `${chunk.start_date} → ${chunk.end_date}`;

  const senderMap: Record<string, string> = {};
  chunk.participants.forEach((p, i) => {
    senderMap[p] = SENDER_COLORS[i % SENDER_COLORS.length];
  });

  const preview = chunk.text.split("\n").slice(0, 2).join(" ").slice(0, 120);

  return (
    <div className="source-card">
      <button className="source-card-header" onClick={() => setOpen((o) => !o)}>
        <div className="source-card-meta">
          <span className="source-rank">#{rank + 1}</span>
          <span className="level-badge">L{chunk.level}</span>
          <span className="source-date">{dateLabel}</span>
          <span className="source-msgs">{chunk.message_count} msgs</span>
          <span className="source-score">{score.toFixed(3)}</span>
        </div>
        <span className="source-preview">{preview}…</span>
        <span className="source-chevron">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="source-card-body">
          <pre className="transcript">
            {colorizeTranscript(chunk.text, senderMap)}
          </pre>
        </div>
      )}
    </div>
  );
}
