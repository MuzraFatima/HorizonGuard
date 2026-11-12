import type { ReactNode } from "react";

export function Skeleton({ h = 16, w = "100%", style }: { h?: number; w?: number | string; style?: React.CSSProperties }) {
  return <div className="skeleton" style={{ height: h, width: w, ...style }} aria-hidden="true" />;
}

export function CardSkeleton({ lines = 3, chart = false }: { lines?: number; chart?: boolean }) {
  return (
    <div className="card" aria-busy="true" aria-label="Loading">
      <Skeleton h={14} w="40%" />
      <div style={{ height: 14 }} />
      {chart && <Skeleton h={180} style={{ marginBottom: 14 }} />}
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} h={12} w={`${85 - i * 12}%`} style={{ marginBottom: 10 }} />
      ))}
    </div>
  );
}

export function Stat({ label, value, note, tone }: { label: string; value: ReactNode; note?: ReactNode; tone?: "good" | "bad" | "warn" }) {
  return (
    <div className="card">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${tone ? `tone-${tone}` : ""}`}>{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

export function Chip({ tone = "muted", children }: { tone?: "good" | "bad" | "warn" | "plum" | "muted"; children: ReactNode }) {
  return <span className={`chip chip-${tone}`}>{children}</span>;
}

export function PageHead({ eyebrow, title, lede, aside }: { eyebrow: string; title: string; lede?: ReactNode; aside?: ReactNode }) {
  return (
    <header className="page-head">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        {lede && <p className="lede">{lede}</p>}
      </div>
      {aside}
    </header>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="error" role="alert">
      Could not load data: {message}. Is <span className="mono">horizonguard serve</span> running?
    </div>
  );
}
