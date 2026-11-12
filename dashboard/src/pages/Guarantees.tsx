import type { Guarantees as GuaranteeData } from "../api";
import { pct } from "../api";
import { Bars, MiniLine } from "../components/charts";
import { CardSkeleton, Chip, ErrorNote, PageHead, Stat } from "../components/ui";
import { useApi } from "../hooks";

export default function Guarantees() {
  const { data, error, loading } = useApi<GuaranteeData>("/api/guarantees");

  return (
    <div className="page">
      <PageHead
        eyebrow="Disruption-controlled guarantees"
        title="How much harm can an alert cause?"
        lede="Most monitors certify how often they fire. HorizonGuard certifies how often its interventions
        break a run that would have succeeded, using fork-labelled calibration runs."
      />
      {error && <ErrorNote message={error} />}
      {loading || !data ? (
        <div className="grid grid-4">
          {[0, 1, 2, 3].map((i) => (
            <CardSkeleton key={i} lines={2} />
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-4">
            <Stat label="Target disruption risk" value={`≤ ${pct(data.certificate.alpha)}`} note="alpha, preregistered" />
            <Stat
              label="Certified threshold"
              value={data.certificate.lambda === null ? "never act" : data.certificate.lambda.toFixed(3)}
              note="minimum predicted net benefit to act"
            />
            <Stat
              label="Calibration runs"
              value={data.certificate.n_runs}
              note={`${data.certificate.method.toUpperCase()} certificate`}
            />
            <Stat
              label="Resamples above target"
              value={pct(data.studies.coverage.ltt_violation_rate)}
              note={`out of ${data.studies.coverage.resamples} resampled splits`}
              tone="good"
            />
          </div>

          <div className="grid grid-2 section">
            <div className="card">
              <div className="card-head">
                <h2>Rescues kept vs calibration size</h2>
                <Chip tone="plum">Learn-then-Test, 90% confidence</Chip>
              </div>
              <MiniLine
                points={data.studies.sample_complexity.map((r) => ({ x: `${r.n} runs`, y: r.retention }))}
                color="#6f4565"
                label="Share of rescues kept by the high-probability certificate"
              />
              <p className="card-sub">
                Below about 40 calibration runs the high-probability certificate is vacuous: it never acts,
                which is safe but useless. From 80 runs it acts, and at 120 it keeps 92% of rescues. It never
                exceeded the target at any size.
              </p>
            </div>
            <div className="card">
              <div className="card-head">
                <h2>Time-uniform false alarms</h2>
                <Chip tone="good">{pct(data.studies.time_uniform.eprocess_fpr)} false-alarm runs</Chip>
              </div>
              <Bars
                rows={[
                  {
                    label: "Betting e-process: detection",
                    value: data.studies.time_uniform.eprocess_detection,
                    color: "#6f4565",
                    note: `mean lead ${data.studies.time_uniform.eprocess_lead.toFixed(1)} steps`,
                  },
                  {
                    label: "Fixed threshold: detection",
                    value: data.studies.time_uniform.fixed_detection,
                    color: "#8a8172",
                    note: `mean lead ${data.studies.time_uniform.fixed_lead.toFixed(1)} steps`,
                  },
                ]}
                max={1}
                format={(v) => pct(v)}
              />
              <p className="card-sub">
                The e-process bounds the chance of any false alarm in a healthy run at 5%, however long the
                run is watched, and still detects slightly more drift than a fixed threshold.
              </p>
            </div>
          </div>

          <section className="section">
            <h2>What is and is not guaranteed</h2>
            <div className="grid grid-2">
              <div className="card">
                <h3 style={{ marginBottom: 8 }}>Guaranteed</h3>
                <ul style={{ margin: 0, paddingLeft: 18, color: "var(--ink-2)" }}>
                  <li>Expected disruption at or below alpha for runs like the calibration runs (CRC).</li>
                  <li>With probability 1 − delta, disruption at or below alpha (Learn-then-Test).</li>
                  <li>At most a 5% chance of any false alarm in a healthy run (e-process, under its assumption).</li>
                </ul>
              </div>
              <div className="card">
                <h3 style={{ marginBottom: 8 }}>Not guaranteed</h3>
                <ul style={{ margin: 0, paddingLeft: 18, color: "var(--ink-2)" }}>
                  <li>Anything for a new agent family until a few of its runs are pooled into calibration.</li>
                  <li>Missed rescues: the certificate bounds harm, not missed opportunities.</li>
                  <li>Agents whose full state cannot be snapshotted, since their fork labels are not unbiased.</li>
                </ul>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
