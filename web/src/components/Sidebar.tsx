import { useEffect, useState } from "react";
import { api } from "../api";
import type { ChatMode, Stats } from "../types";

interface Props {
  mode: ChatMode;
  onModeChange: (m: ChatMode) => void;
  topK: number;
  onTopKChange: (k: number) => void;
}

export function Sidebar({ mode, onModeChange, topK, onTopKChange }: Props) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .stats()
      .then(setStats)
      .catch((e: Error) => setError(e.message));
  }, []);

  const totalSenders = stats
    ? Object.values(stats.senders).reduce((a, b) => a + b, 0)
    : 0;

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="logo-icon">📱</span>
        <span className="logo-text">Instagram RAG</span>
      </div>

      <div className="sidebar-section">
        <p className="section-label">Mode</p>
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === "ask" ? "active" : ""}`}
            onClick={() => onModeChange("ask")}
          >
            Ask
          </button>
          <button
            className={`mode-btn ${mode === "search" ? "active" : ""}`}
            onClick={() => onModeChange("search")}
          >
            Search
          </button>
        </div>
        <p className="mode-desc">
          {mode === "ask"
            ? "Claude reads the retrieved messages and writes an answer."
            : "Returns matching message chunks directly — no AI, no synthesis."}
        </p>
      </div>

      <div className="sidebar-section">
        <p className="section-label">Results (top-{topK})</p>
        <div className="slider-row">
          <input
            type="range"
            min={3}
            max={20}
            value={topK}
            onChange={(e) => onTopKChange(Number(e.target.value))}
            className="slider"
          />
          <span className="slider-value">{topK}</span>
        </div>
      </div>

      <div className="sidebar-section">
        <p className="section-label">Index</p>
        {error ? (
          <p className="stat-error">{error}</p>
        ) : stats ? (
          <div className="stats">
            <div className="stat-row">
              <span className="stat-label">Messages</span>
              <span className="stat-value">{stats.message_count.toLocaleString()}</span>
            </div>
            {stats.date_range && (
              <div className="stat-row">
                <span className="stat-label">Range</span>
                <span className="stat-value small">
                  {stats.date_range.start}
                  <br />
                  {stats.date_range.end}
                </span>
              </div>
            )}
            {stats.chroma.l2 != null && (
              <>
                <div className="stat-row">
                  <span className="stat-label">L2 chunks</span>
                  <span className="stat-value">{stats.chroma.l2.toLocaleString()}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">L3 segments</span>
                  <span className="stat-value">{stats.chroma.l3?.toLocaleString() ?? "—"}</span>
                </div>
              </>
            )}
            {Object.entries(stats.senders).length > 0 && (
              <div className="senders">
                <p className="section-label" style={{ marginTop: 12 }}>Participants</p>
                {Object.entries(stats.senders)
                  .sort((a, b) => b[1] - a[1])
                  .map(([sender, count]) => (
                    <div key={sender} className="sender-row">
                      <span className="sender-name">{sender}</span>
                      <div className="sender-bar-wrap">
                        <div
                          className="sender-bar"
                          style={{ width: `${(count / totalSenders) * 100}%` }}
                        />
                      </div>
                      <span className="sender-pct">
                        {((count / totalSenders) * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
              </div>
            )}
          </div>
        ) : (
          <p className="stat-loading">Loading…</p>
        )}
      </div>
    </aside>
  );
}
