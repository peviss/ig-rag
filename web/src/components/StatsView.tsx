import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import type { MessageStats } from "../types";

// Colors assigned per sender index
const SENDER_COLORS = ["#818cf8", "#f472b6", "#34d399", "#fbbf24", "#60a5fa"];

const CHART_THEME = {
  background: "transparent",
  text: "#6060808",
  grid: "#1e1e30",
  tooltip: { bg: "#1a1a26", border: "#252535", text: "#e0e0f0" },
};

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: CHART_THEME.tooltip.bg,
        border: `1px solid ${CHART_THEME.tooltip.border}`,
        borderRadius: 8,
        padding: "8px 12px",
        fontSize: "0.8rem",
        color: CHART_THEME.tooltip.text,
      }}
    >
      <p style={{ fontWeight: 600, marginBottom: 4 }}>{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value.toLocaleString()}
        </p>
      ))}
    </div>
  );
}

function StatCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
}) {
  return (
    <div className="stat-card">
      <p className="stat-card-label">{label}</p>
      <p className="stat-card-value" style={color ? { color } : undefined}>
        {typeof value === "number" ? value.toLocaleString() : value}
      </p>
      {sub && <p className="stat-card-sub">{sub}</p>}
    </div>
  );
}

export function StatsView() {
  const [data, setData] = useState<MessageStats | null>(null);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    load(start, end);
  }, []);

  async function load(s: string, e: string) {
    setLoading(true);
    setError(null);
    try {
      const result = await api.messageStats(s || undefined, e || undefined);
      setData(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  function handleFilter() {
    load(start, end);
  }

  function handleClear() {
    setStart("");
    setEnd("");
    load("", "");
  }

  if (error) {
    return <div className="stats-view-error">{error}</div>;
  }

  const senders = data?.senders ?? [];

  return (
    <div className="stats-view">
      {/* Date filter */}
      <div className="stats-filter-bar">
        <label className="filter-label">From</label>
        <input
          type="date"
          className="date-input"
          value={start}
          onChange={(e) => setStart(e.target.value)}
        />
        <label className="filter-label">To</label>
        <input
          type="date"
          className="date-input"
          value={end}
          onChange={(e) => setEnd(e.target.value)}
        />
        <button className="filter-btn" onClick={handleFilter} disabled={loading}>
          Apply
        </button>
        {(start || end) && (
          <button className="filter-btn secondary" onClick={handleClear}>
            Clear
          </button>
        )}
        {loading && <span className="filter-loading">Loading…</span>}
        {data && !loading && (
          <span className="filter-total">{data.total.toLocaleString()} messages</span>
        )}
      </div>

      {data && (
        <div className="stats-scroll">
          {/* ── Summary row ─────────────────────────────────────────────────── */}
          <section className="stats-section">
            <h2 className="stats-section-title">Overview</h2>
            <div className="stat-cards-row">
              <StatCard label="Total messages" value={data.total} />
              <StatCard label="Active days" value={data.summary.active_days}
                sub={`of ${data.summary.total_days} total`} />
              <StatCard label="Avg / day" value={data.summary.avg_per_day} />
              <StatCard label="Longest streak" value={`${data.summary.longest_streak}d`}
                color="#34d399" sub="consecutive days" />
              <StatCard label="Longest gap" value={`${data.summary.longest_gap}d`}
                color="#f87171" sub="no messages" />
              <StatCard label="First message" value={data.summary.first_date} />
              <StatCard label="Last message" value={data.summary.last_date} />
            </div>
          </section>

          {/* ── Per-sender ───────────────────────────────────────────────────── */}
          <section className="stats-section">
            <h2 className="stats-section-title">By participant</h2>
            <div className="sender-cards-row">
              {senders.map((sender, i) => {
                const s = data.by_sender[sender];
                const color = SENDER_COLORS[i % SENDER_COLORS.length];
                return (
                  <div key={sender} className="sender-stat-card" style={{ borderTopColor: color }}>
                    <p className="sender-stat-name" style={{ color }}>{sender}</p>
                    <div className="sender-stat-grid">
                      <div>
                        <p className="sender-stat-val">{s.count.toLocaleString()}</p>
                        <p className="sender-stat-key">messages</p>
                      </div>
                      <div>
                        <p className="sender-stat-val">{s.pct}%</p>
                        <p className="sender-stat-key">of total</p>
                      </div>
                      <div>
                        <p className="sender-stat-val">{s.avg_length}</p>
                        <p className="sender-stat-key">avg chars</p>
                      </div>
                      <div>
                        <p className="sender-stat-val">{s.media_count.toLocaleString()}</p>
                        <p className="sender-stat-key">media sent</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* ── Charts row ───────────────────────────────────────────────────── */}
          <div className="charts-row">
            {/* By hour */}
            <section className="stats-section chart-section">
              <h2 className="stats-section-title">Activity by hour <span className="utc-note">(UTC−3)</span></h2>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={data.by_hour} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} vertical={false} />
                  <XAxis dataKey="hour" tick={{ fill: "#55556a", fontSize: 11 }}
                    tickFormatter={(h) => `${h}h`} />
                  <YAxis tick={{ fill: "#55556a", fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: "0.75rem" }} />
                  {senders.map((s, i) => (
                    <Bar key={s} dataKey={s} stackId="a"
                      fill={SENDER_COLORS[i % SENDER_COLORS.length]} radius={i === senders.length - 1 ? [3, 3, 0, 0] : undefined} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </section>

            {/* By weekday */}
            <section className="stats-section chart-section">
              <h2 className="stats-section-title">Activity by day of week</h2>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={data.by_weekday} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} vertical={false} />
                  <XAxis dataKey="day" tick={{ fill: "#55556a", fontSize: 11 }} />
                  <YAxis tick={{ fill: "#55556a", fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: "0.75rem" }} />
                  {senders.map((s, i) => (
                    <Bar key={s} dataKey={s} stackId="a"
                      fill={SENDER_COLORS[i % SENDER_COLORS.length]} radius={i === senders.length - 1 ? [3, 3, 0, 0] : undefined} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </section>
          </div>

          {/* ── Monthly trend ─────────────────────────────────────────────────── */}
          <section className="stats-section">
            <h2 className="stats-section-title">Messages per month</h2>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={data.by_month} margin={{ top: 4, right: 16, left: -16, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} vertical={false} />
                <XAxis dataKey="month" tick={{ fill: "#55556a", fontSize: 11 }}
                  tickFormatter={(m) => m.slice(2)} />
                <YAxis tick={{ fill: "#55556a", fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: "0.75rem" }} />
                {senders.map((s, i) => (
                  <Line key={s} type="monotone" dataKey={s}
                    stroke={SENDER_COLORS[i % SENDER_COLORS.length]}
                    strokeWidth={2} dot={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </section>

          {/* ── Top days ──────────────────────────────────────────────────────── */}
          <section className="stats-section">
            <h2 className="stats-section-title">Most active days</h2>
            <div className="top-days-list">
              {data.top_days.map((row, i) => {
                const date = row.date as string;
                const count = row.count as number;
                const max = (data.top_days[0].count as number);
                return (
                  <div key={date} className="top-day-row">
                    <span className="top-day-rank">#{i + 1}</span>
                    <span className="top-day-date">{date}</span>
                    <div className="top-day-bar-wrap">
                      <div className="top-day-bar" style={{ width: `${(count / max) * 100}%` }} />
                      <div className="top-day-segments">
                        {senders.map((s, si) => {
                          const sc = row[s] as number;
                          return (
                            <span key={s} className="top-day-sender-count"
                              style={{ color: SENDER_COLORS[si % SENDER_COLORS.length] }}>
                              {sc}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                    <span className="top-day-total">{count.toLocaleString()}</span>
                  </div>
                );
              })}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
