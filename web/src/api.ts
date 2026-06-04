import type { AskResponse, MessageStats, SearchResponse, Stats } from "./types";

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
};
