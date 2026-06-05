import type { AskResponse, Config, IngestEvent, MessageStats, SearchResponse, Stats } from "./types";

const BASE = "/api";

async function post<T>(path: string, body: object): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

export const api = {
  ask: (query: string, top_k = 8) =>
    post<AskResponse>("/ask", { query, top_k }),

  search: (query: string, k = 10) =>
    post<SearchResponse>("/search", { query, k }),

  stats: () => get<Stats>("/stats"),

  messageStats: (start?: string, end?: string, tzOffset = -3) => {
    const params = new URLSearchParams({ tz_offset: String(tzOffset) });
    if (start) params.set("start", start);
    if (end) params.set("end", end);
    return get<MessageStats>(`/stats/messages?${params}`);
  },

  health: () => get<{ status: string; index_loaded: boolean }>("/health"),

  getConfig: () => get<Config>("/config"),

  setConfig: (config: Config) => post<Config>("/config", config),

  ingest: async (
    req: { inbox_path: string; model?: string; force?: boolean },
    onProgress: (e: IngestEvent) => void,
  ): Promise<void> => {
    const res = await fetch(`${BASE}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail ?? res.statusText);
    }
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            onProgress(JSON.parse(line.slice(6)));
          } catch {
            // ignore malformed lines
          }
        }
      }
    }
  },
};
