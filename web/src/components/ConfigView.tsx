import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Config, IngestEvent } from "../types";

export function ConfigView() {
  const [config, setConfig] = useState<Config>({ inbox_path: "./messages/inbox", model: "paraphrase-multilingual-MiniLM-L12-v2" });
  const [saved, setSaved] = useState(false);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<IngestEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.getConfig().then(setConfig).catch(() => {});
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log]);

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await api.setConfig(config);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleIngest() {
    setError(null);
    setLog([]);
    setRunning(true);
    try {
      await api.ingest({ inbox_path: config.inbox_path, model: config.model }, (event) => {
        setLog((prev) => [...prev, event]);
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  }

  const lastEvent = log[log.length - 1];
  const progress = lastEvent?.step && lastEvent?.total
    ? Math.round((lastEvent.step / lastEvent.total) * 100)
    : 0;

  return (
    <div className="config-view">
      <div className="config-scroll">
        <section className="config-section">
          <h2 className="config-section-title">Data Source</h2>
          <form className="config-form" onSubmit={handleSave}>
            <div className="config-field">
              <label className="config-label">Instagram inbox path</label>
              <p className="config-hint">
                Absolute or relative path to the <code>messages/inbox</code> directory from your Instagram export.
              </p>
              <input
                className="config-input"
                type="text"
                value={config.inbox_path}
                onChange={(e) => setConfig({ ...config, inbox_path: e.target.value })}
                placeholder="./messages/inbox"
                disabled={running}
              />
            </div>

            <div className="config-field">
              <label className="config-label">Embedding model</label>
              <p className="config-hint">Sentence-transformers model name (multilingual recommended).</p>
              <input
                className="config-input"
                type="text"
                value={config.model}
                onChange={(e) => setConfig({ ...config, model: e.target.value })}
                placeholder="paraphrase-multilingual-MiniLM-L12-v2"
                disabled={running}
              />
            </div>

            {error && <p className="config-error">{error}</p>}

            <div className="config-actions">
              <button className="config-btn secondary" type="submit" disabled={running}>
                {saved ? "Saved!" : "Save settings"}
              </button>
              <button
                className="config-btn primary"
                type="button"
                onClick={handleIngest}
                disabled={running}
              >
                {running ? (
                  <>
                    <span className="btn-spinner" />
                    Generating…
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                    </svg>
                    Generate embeddings
                  </>
                )}
              </button>
            </div>
          </form>
        </section>

        {(log.length > 0 || running) && (
          <section className="config-section">
            <h2 className="config-section-title">
              Ingestion progress
              {running && <span className="ingest-running-badge">running</span>}
              {lastEvent?.done && <span className="ingest-done-badge">done</span>}
            </h2>

            {running && lastEvent?.step != null && (
              <div className="ingest-progress-wrap">
                <div className="ingest-progress-bar" style={{ width: `${progress}%` }} />
                <span className="ingest-progress-label">Step {lastEvent.step} / {lastEvent.total}</span>
              </div>
            )}

            <div className="ingest-log">
              {log.map((ev, i) => (
                <div
                  key={i}
                  className={`ingest-log-line ${ev.error ? "error" : ev.done ? "done" : ""}`}
                >
                  {ev.step != null && (
                    <span className="ingest-step-badge">{ev.step}/{ev.total}</span>
                  )}
                  <span className="ingest-log-msg">{ev.message}</span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>

            {lastEvent?.done && (
              <div className="ingest-summary">
                <div className="ingest-summary-item">
                  <span className="ingest-summary-val">{lastEvent.message_count?.toLocaleString()}</span>
                  <span className="ingest-summary-key">messages</span>
                </div>
                <div className="ingest-summary-item">
                  <span className="ingest-summary-val">{lastEvent.l2?.toLocaleString()}</span>
                  <span className="ingest-summary-key">L2 chunks</span>
                </div>
                <div className="ingest-summary-item">
                  <span className="ingest-summary-val">{lastEvent.l3?.toLocaleString()}</span>
                  <span className="ingest-summary-key">L3 segments</span>
                </div>
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
