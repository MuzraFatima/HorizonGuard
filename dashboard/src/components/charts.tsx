import { memo, useMemo } from "react";
import type { Step } from "../api";

const W = 760;
const H = 230;
const PAD = { l: 40, r: 14, t: 14, b: 28 };

interface TimelineProps {
  steps: Step[];
  cursor: number;
  violationStep: number | null;
}

/** Risk, fidelity and drifting belief over a run, with interventions and violation marked. */
export const TimelineChart = memo(function TimelineChart({ steps, cursor, violationStep }: TimelineProps) {
  const geo = useMemo(() => {
    const n = Math.max(steps.length - 1, 1);
    const x = (t: number) => PAD.l + (t / n) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - v) * (H - PAD.t - PAD.b);
    const line = (key: "risk" | "fidelity" | "p_drifting") =>
      steps.map((s, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(s[key]).toFixed(1)}`).join("");
    const band =
      steps.map((s, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(s.p_drifting).toFixed(1)}`).join("") +
      `L${x(steps.length - 1).toFixed(1)},${y(0)}L${x(0)},${y(0)}Z`;
    return { x, y, risk: line("risk"), fidelity: line("fidelity"), band };
  }, [steps]);

  if (!steps.length) return null;
  const { x, y } = geo;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const cur = Math.min(cursor, steps.length - 1);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Risk and fidelity over the run">
      {ticks.map((v) => (
        <g key={v}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="#ece4d4" />
          <text x={PAD.l - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="#6f685c" fontFamily="IBM Plex Mono">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      <path d={geo.band} fill="#ecdde7" opacity="0.55" />
      <path d={geo.fidelity} fill="none" stroke="#4f6f41" strokeWidth="2" />
      <path d={geo.risk} fill="none" stroke="#b4492a" strokeWidth="2.4" />
      {steps.map((s, i) =>
        s.applied ? (
          <g key={`a${i}`}>
            <line x1={x(i)} x2={x(i)} y1={PAD.t} y2={H - PAD.b} stroke="#b97d12" strokeDasharray="3 3" />
            <circle cx={x(i)} cy={PAD.t + 4} r="4.5" fill="#e0a83b" stroke="#fffdf8" strokeWidth="1.5" />
          </g>
        ) : null,
      )}
      {violationStep !== null && violationStep < steps.length && (
        <g>
          <rect x={x(violationStep) - 1} y={PAD.t} width="2.5" height={H - PAD.t - PAD.b} fill="#b4492a" />
          <text x={x(violationStep) + 6} y={PAD.t + 12} fontSize="11.5" fill="#8e3820" fontWeight="600">
            violation
          </text>
        </g>
      )}
      <line x1={x(cur)} x2={x(cur)} y1={PAD.t} y2={H - PAD.b} stroke="#22201c" strokeWidth="1" opacity="0.5" />
      <circle cx={x(cur)} cy={y(steps[cur].risk)} r="4.5" fill="#b4492a" stroke="#fffdf8" strokeWidth="2" />
      <text x={PAD.l} y={H - 8} fontSize="11" fill="#6f685c">step 0</text>
      <text x={W - PAD.r} y={H - 8} fontSize="11" fill="#6f685c" textAnchor="end">
        step {steps.length - 1}
      </text>
    </svg>
  );
});

interface BarsProps {
  rows: { label: string; value: number; color: string; note?: string }[];
  max?: number;
  format?: (v: number) => string;
}

/** Horizontal bars with labels and values; used for A/B comparisons. */
export const Bars = memo(function Bars({ rows, max, format = (v) => v.toFixed(2) }: BarsProps) {
  const top = max ?? Math.max(...rows.map((r) => r.value), 1e-9);
  return (
    <div>
      {rows.map((r) => (
        <div key={r.label} style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13.5, marginBottom: 5 }}>
            <span style={{ fontWeight: 500 }}>{r.label}</span>
            <span className="mono">{format(r.value)}</span>
          </div>
          <div className="arm-track" style={{ height: 12 }}>
            <div className="arm-bar" style={{ left: 0, width: `${(r.value / top) * 100}%`, background: r.color }} />
          </div>
          {r.note && <div className="card-sub" style={{ marginTop: 4 }}>{r.note}</div>}
        </div>
      ))}
    </div>
  );
});

/** A small line chart for a single series over labelled x points. */
export const MiniLine = memo(function MiniLine({
  points,
  color,
  label,
  yMax = 1,
}: {
  points: { x: string; y: number }[];
  color: string;
  label: string;
  yMax?: number;
}) {
  const w = 360;
  const h = 150;
  const p = { l: 34, r: 34, t: 10, b: 24 }; // room for centred end labels
  const xs = (i: number) => p.l + (i / Math.max(points.length - 1, 1)) * (w - p.l - p.r);
  const ys = (v: number) => p.t + (1 - v / yMax) * (h - p.t - p.b);
  const d = points.map((pt, i) => `${i ? "L" : "M"}${xs(i)},${ys(pt.y)}`).join("");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img" aria-label={label}>
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={p.l} x2={w - p.r} y1={ys(v * yMax)} y2={ys(v * yMax)} stroke="#ece4d4" />
          <text x={p.l - 6} y={ys(v * yMax) + 4} textAnchor="end" fontSize="10.5" fill="#6f685c" fontFamily="IBM Plex Mono">
            {(v * yMax).toFixed(1)}
          </text>
        </g>
      ))}
      <path d={d} fill="none" stroke={color} strokeWidth="2.4" />
      {points.map((pt, i) => (
        <g key={pt.x}>
          <circle cx={xs(i)} cy={ys(pt.y)} r="4" fill={color} stroke="#fffdf8" strokeWidth="1.5" />
          <text x={xs(i)} y={h - 6} textAnchor="middle" fontSize="11" fill="#4f4a41">
            {pt.x}
          </text>
        </g>
      ))}
    </svg>
  );
});
