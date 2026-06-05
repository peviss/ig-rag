export interface Chunk {
  chunk_id: string;
  level: 1 | 2 | 3;
  thread_id: string;
  participants: string[];
  text: string;
  message_ids: string[];
  start_timestamp: number;
  end_timestamp: number;
  start_date: string;
  end_date: string;
  year: number;
  month: number;
  topic: string | null;
  message_count: number;
  sender: string | null;
}

export interface Source {
  rank: number;
  score: number;
  chunk_id: string;
  chunk: Chunk;
}

export interface Intent {
  type: "search" | "summarize" | "timeline" | "count" | "comparison";
  entities: string[];
  temporal_anchor: string | null;
  date_start: number | null;
  date_end: number | null;
  expanded_queries: string[];
}

export interface AskResponse {
  answer: string;
  has_llm_answer: boolean;
  query: string;
  intent: Intent | null;
  sources: Source[];
}

export interface SearchResponse {
  query: string;
  intent: Intent;
  results: Source[];
}

export interface Stats {
  message_count: number;
  bm25_size: number;
  date_range: { start: string; end: string } | null;
  senders: Record<string, number>;
  chroma: { l2?: number; l3?: number };
  index_ready: boolean;
}

export type ChatMode = "ask" | "search";

// ── Stats ──────────────────────────────────────────────────────────────────────

export interface SenderStats {
  count: number;
  avg_length: number;
  media_count: number;
  pct: number;
}

export interface StatSummary {
  first_date: string;
  last_date: string;
  active_days: number;
  total_days: number;
  avg_per_day: number;
  longest_streak: number;
  longest_gap: number;
}

export interface MessageStats {
  total: number;
  senders: string[];
  by_sender: Record<string, SenderStats>;
  /** 24 items, one per hour (UTC) */
  by_hour: Array<Record<string, number>>;
  /** 7 items Mon–Sun */
  by_weekday: Array<Record<string, number>>;
  by_month: Array<Record<string, number>>;
  top_days: Array<Record<string, number | string>>;
  summary: StatSummary;
}

export interface Config {
  inbox_path: string;
  model: string;
}

export interface IngestEvent {
  step?: number;
  total?: number;
  message: string;
  done?: boolean;
  error?: boolean;
  l2?: number;
  l3?: number;
  message_count?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  intent?: Intent | null;
  has_llm_answer?: boolean;
  loading?: boolean;
}
