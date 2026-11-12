import { Link } from "react-router-dom";
import type { Overview as OverviewData } from "../api";
import { pct } from "../api";
import { Bars } from "../components/charts";
import { CardSkeleton, Chip, ErrorNote, PageHead, Stat } from "../components/ui";
import { useApi } from "../hooks";
import { POLICY_COLOR, POLICY_LABEL } from "../labels";

const VERDICT_TONE = { supported: "good", "not supported": "bad", "not met": "warn" } as const;

export default function Overview() {
  const { data, error, loading } = useApi<OverviewData>("/api/overview");

  return (
    <div className="page">
      <PageHead
        eyebrow="Recoverability-aware early warning"
        title="Is the agent still fixable, and will fixing it help?"
        lede="HorizonGuard watches long-horizon agents, forecasts goal drift, and only steps in when the
        predicted rescue outweighs the risk of breaking a run that was going to succeed, with that risk
        certified on counterfactual forks."
      />
      {error && <ErrorNote message={error} />}

      {loading || !data ? (
        <div className="grid grid-4">
          {Array.from({ length: 4 }, (_, i) => (
            <CardSkeleton key={i} lines={2} />
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-4">
            <Stat
              label="Violations, no monitor"
              value={pct(data.ab.none.violation_rate)}
              note="fresh environment and agent family"
              tone="bad"
            />
            <Stat
              label="Violations, probability alarm"
              value={pct(data.ab.probability.violation_rate)}
              note={`${data.ab.probability.rescued} rescued · ${data.ab.probability.disrupted} disrupted`}
              tone="warn"
            />
            <Stat
              label="Violations, HorizonGuard"
              value={pct(data.ab.horizonguard.violation_rate)}
              note={`${data.ab.horizonguard.rescued} rescued · ${data.ab.horizonguard.disrupted} disrupted`}
              tone="good"
            />
            <Stat
              label="Certified disruption risk"
              value={`≤${pct(data.certificate.alpha)}`}
              note={`${data.certificate.method.toUpperCase()} on ${data.certificate.n_runs} calibration runs`}
            />
          </div>

          <div className="split section">
            <div className="card">
              <div className="card-head">
                <h2>Closed-loop A/B</h2>
                <Link to="/ab" className="card-sub">
                  see every seed →
                </Link>
              </div>
              <p className="card-sub" style={{ marginTop: -6, marginBottom: 16 }}>
                Same seeds in every arm, on a support-desk environment and an agent family neither model saw
                in training.
              </p>
              <Bars
                rows={(["none", "probability", "horizonguard"] as const).map((p) => ({
                  label: POLICY_LABEL[p],
                  value: data.ab[p].violation_rate,
                  color: POLICY_COLOR[p],
                  note: `${data.ab[p].interventions_per_run.toFixed(1)} interventions per run`,
                }))}
                max={1}
                format={(v) => pct(v)}
              />
            </div>
            <div className="card">
              <h2 style={{ marginBottom: 12 }}>How to read this dashboard</h2>
              <ol style={{ margin: 0, paddingLeft: 18, color: "var(--ink-2)", fontSize: 14.5 }}>
                <li style={{ marginBottom: 8 }}>
                  <strong>Live monitor</strong> runs an agent step by step and shows what HorizonGuard sees
                  and decides.
                </li>
                <li style={{ marginBottom: 8 }}>
                  <strong>Runs</strong> is the library of closed-loop episodes behind the A/B result.
                </li>
                <li style={{ marginBottom: 8 }}>
                  <strong>Guarantees</strong> explains the disruption certificate and its limits.
                </li>
                <li>
                  <strong>Method</strong> shows how forks turn "would it have helped?" into a measurable
                  label.
                </li>
              </ol>
              <div className="callout" style={{ marginTop: 16 }}>
                All agents here are simulated. Results describe the method, not any real model.
              </div>
            </div>
          </div>

          <section className="section">
            <h2>Research questions and verdicts</h2>
            <div className="card" style={{ padding: "16px 10px 6px" }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>RQ</th>
                    <th>Claim</th>
                    <th>Verdict</th>
                    <th>Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {data.findings.map((f) => (
                    <tr key={f.claim}>
                      <td className="mono">{f.rq}</td>
                      <td>{f.claim}</td>
                      <td>
                        <Chip tone={VERDICT_TONE[f.verdict]}>{f.verdict}</Chip>
                      </td>
                      <td style={{ color: "var(--ink-2)" }}>{f.evidence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
