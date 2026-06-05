import { lazy, Suspense, useState } from "react";
import { ChatView } from "./components/ChatView";
import { ConfigView } from "./components/ConfigView";
import { Sidebar } from "./components/Sidebar";
import type { ChatMode } from "./types";
import "./App.css";

const StatsView = lazy(() =>
  import("./components/StatsView").then((m) => ({ default: m.StatsView }))
);

type View = "chat" | "stats" | "config";

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [mode, setMode] = useState<ChatMode>("ask");
  const [topK, setTopK] = useState(8);

  return (
    <div className="app">
      <Sidebar
        mode={mode}
        onModeChange={setMode}
        topK={topK}
        onTopKChange={setTopK}
      />
      <main className="main">
        <nav className="view-nav">
          <button
            className={`view-tab ${view === "chat" ? "active" : ""}`}
            onClick={() => setView("chat")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            Chat
          </button>
          <button
            className={`view-tab ${view === "stats" ? "active" : ""}`}
            onClick={() => setView("stats")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="18" y1="20" x2="18" y2="10" />
              <line x1="12" y1="20" x2="12" y2="4" />
              <line x1="6" y1="20" x2="6" y2="14" />
            </svg>
            Dashboard
          </button>
          <button
            className={`view-tab ${view === "config" ? "active" : ""}`}
            onClick={() => setView("config")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            Config
          </button>
        </nav>

        {view === "chat" ? (
          <ChatView mode={mode} topK={topK} />
        ) : view === "stats" ? (
          <Suspense fallback={<div className="stats-loading-fallback">Loading…</div>}>
            <StatsView />
          </Suspense>
        ) : (
          <ConfigView />
        )}
      </main>
    </div>
  );
}
