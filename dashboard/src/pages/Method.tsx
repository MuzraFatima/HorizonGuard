import { PageHead } from "../components/ui";

const STAGES = [
  ["Agent", "tools, memory, plans"],
  ["Trajectory log", "actions, observations, reasoning"],
  ["Embedding", "goal-relative geometry"],
  ["Hazard model", "aligned, drifting, violated"],
  ["Effect model", "rescue vs disruption per strength"],
  ["Certificate", "disruption risk ≤ alpha"],
  ["Policy", "re-anchor, gate, escalate or wait"],
];

function Pipeline() {
  const w = 1000;
  const box = 128;
  const gap = (w - STAGES.length * box) / (STAGES.length - 1);
  return (
    <svg viewBox={`0 0 ${w} 190`} width="100%" role="img" aria-label="Monitoring pipeline">
      {STAGES.map(([title, sub], i) => {
        const x = i * (box + gap);
        return (
          <g key={title}>
            {i > 0 && (
              <path
                d={`M${x - gap + 4},55 L${x - 8},55`}
                stroke="#b97d12"
                strokeWidth="2"
                markerEnd="url(#arrow)"
              />
            )}
            <rect x={x} y={20} width={box} height={70} rx={12} fill="#fffdf8" stroke="#d3c8b2" />
            <text x={x + box / 2} y={50} textAnchor="middle" fontSize="14" fontWeight="600" fill="#22201c">
              {title}
            </text>
            <text x={x + box / 2} y={70} textAnchor="middle" fontSize="11" fill="#6f685c">
              {sub}
            </text>
          </g>
        );
      })}
      <rect x={0} y={128} width={w} height={46} rx={12} fill="#ecdde7" />
      <text x={w / 2} y={156} textAnchor="middle" fontSize="13.5" fill="#4d2f46">
        Fork engine (offline only): snapshot a prefix and branch K continuations with and without each
        intervention, giving risk and effect labels
      </text>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#b97d12" />
        </marker>
      </defs>
    </svg>
  );
}

export default function Method() {
  return (
    <div className="page">
      <PageHead
        eyebrow="Method"
        title="From 'will it drift?' to 'will fixing it help?'"
        lede="The monitor reads only observable behaviour. Training labels come from forking: replay a run to a
        step, then branch seed-matched continuations with and without each intervention."
      />
      <div className="card">
        <Pipeline />
      </div>
      <div className="grid grid-3 section">
        <div className="card">
          <h3 style={{ marginBottom: 8 }}>1. Fork labels</h3>
          <p className="card-sub" style={{ fontSize: 14 }}>
            Because every arm shares the same seeds, each seed shows both potential outcomes. A rescue
            (fails without, succeeds with) and a disruption (the reverse) become directly measurable.
          </p>
        </div>
        <div className="card">
          <h3 style={{ marginBottom: 8 }}>2. Effect-aware decisions</h3>
          <p className="card-sub" style={{ fontSize: 14 }}>
            For every strength, from a one-line reminder to escalation, the model predicts rescue and
            disruption, then picks the intervention with the best rescue − 2 × disruption, or waits.
          </p>
        </div>
        <div className="card">
          <h3 style={{ marginBottom: 8 }}>3. Certified harm</h3>
          <p className="card-sub" style={{ fontSize: 14 }}>
            The action threshold is set by conformal risk control on whole calibration runs, so the expected
            share of decisions that break a healthy run stays at or below alpha.
          </p>
        </div>
      </div>
      <div className="callout section">
        Reproduce everything with <span className="mono">make all</span>. Every number on this dashboard comes
        from the generated DriftBench-CF benchmark and the closed-loop A/B.
      </div>
    </div>
  );
}
