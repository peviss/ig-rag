import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Config, IngestEvent } from "../types";

export function ConfigView() {
  const [config, setConfig] = useState<Config>({ inbox_path: "", model: "paraphrase-multilingual-MiniLM-L12-v2" });
  const [saved, setSaved] = useState(false);
  const [picking, setPicking] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [uploadResult, setUploadResult] = useState<{ threads: number; s3_key: string | null } | null>(null);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<IngestEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getConfig().then(setConfig).catch(() => {});
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log]);

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setUploadResult(null);
    setUploading(true);
    setUploadPct(0);
    try {
      const result = await api.uploadInbox(file, setUploadPct);
      setConfig((prev) => ({ ...prev, inbox_path: result.inbox_path }));
      setUploadResult({ threads: result.threads, s3_key: result.s3_key });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handlePickFolder() {
    setPicking(true);
    setError(null);
    try {
      const result = await api.pickFolder();
      if (!result.cancelled && result.path) {
        setConfig((prev) => ({ ...prev, inbox_path: result.path! }));
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPicking(false);
    }
  }

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
              <label className="config-label">Instagram inbox</label>
              <p className="config-hint">
                Upload your Instagram export <code>.zip</code>, or browse to the extracted <code>messages/inbox</code> folder.
              </p>

              <input
                ref={fileInputRef}
                type="file"
                accept=".zip"
                style={{ display: "none" }}
                onChange={handleFileChange}
              />

              <div className="folder-picker-row">
                <div className="folder-picker-display">
                  {config.inbox_path ? (
                    <span className="folder-path">{config.inbox_path}</span>
                  ) : (
                    <span className="folder-path-empty">No folder selected</span>
                  )}
                </div>
                <button
                  className="config-btn secondary folder-btn"
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={running || picking || uploading}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polyline points="16 16 12 12 8 16" />
                    <line x1="12" y1="12" x2="12" y2="21" />
                    <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
                  </svg>
                  Upload ZIP
                </button>
                <button
                  className="config-btn secondary folder-btn"
                  type="button"
                  onClick={handlePickFolder}
                  disabled={running || picking || uploading}
                >
                  {picking ? (
                    <><span className="btn-spinner dark" />Waiting…</>
                  ) : (
                    <>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                      </svg>
                      Browse…
                    </>
                  )}
                </button>
              </div>

              {uploading && (
                <div className="upload-progress-wrap">
                  <div className="upload-progress-bar" style={{ width: `${uploadPct}%` }} />
                  <span className="upload-progress-label">{uploadPct}%</span>
                </div>
              )}

              {uploadResult && !uploading && (
                <div className="upload-success">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  Extracted {uploadResult.threads} conversation{uploadResult.threads !== 1 ? "s" : ""}
                  {uploadResult.s3_key && <span className="upload-s3-note"> · saved to S3</span>}
                </div>
              )}
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
