import { useNavigate } from "react-router-dom";
import type { ABRow, ArmSummary, Policy } from "../api";
import { pct } from "../api";
import { Bars } from "../components/charts";
import { CardSkeleton, ErrorNote, PageHead, Stat } from "../components/ui";
import { useApi } from "../hooks";
import { POLICY_COLOR, POLICY_LABEL } from "../labels";

const ARMS: Policy[] = ["none", "probability", "horizonguard"];

interface ABData {
  summary: Record<Policy, ArmSummary>;
  rows: ABRow[];
}

export default function ABTest() {
  const { data, error, loading } = useApi<ABData>("/api/ab");
  const navigate = useNavigate();

  return (
    <div className="page">
      <PageHead
        eyebrow="Closed-loop validation"
        title="Fork-controlled A/B on unseen agents"
        lede="Each seed is played three times with identical randomness, so a run that violates without the
        monitor but not with it is a rescue, and the reverse is a disruption. Nothing here was used to train
        or calibrate the monitor."
      />
      {error && <ErrorNote message={error} />}
      {loading || !data ? (
        <div className="grid grid-3">
          {[0, 1, 2].map((i) => (
            <CardSkeleton key={i} lines={3} />
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-3">
            {ARMS.map((p) => (
              <Stat
                key={p}
                label={POLICY_LABEL[p]}
                value={pct(data.summary[p].violation_rate)}
                note={
                  p === "none"
                    ? "violation rate without help"
                    : `${data.summary[p].rescued} rescued · ${data.summary[p].disrupted} disrupted`
                }
                tone={p === "none" ? "bad" : p === "probability" ? "warn" : "good"}
              />
            ))}
          </div>

          <div className="grid grid-2 section">
            <div className="card">
              <h2 style={{ marginBottom: 16 }}>Rescues vs disruptions</h2>
              <Bars
                rows={ARMS.filter((p) => p !== "none").flatMap((p) => [
                  { label: `${POLICY_LABEL[p]}: rescued`, value: data.summary[p].rescued, color: "#4f6f41" },
                  { label: `${POLICY_LABEL[p]}: disrupted`, value: data.summary[p].disrupted, color: "#b4492a" },
                ])}
                format={(v) => String(Math.round(v))}
              />
            </div>
            <div className="card">
              <h2 style={{ marginBottom: 16 }}>Interventions per run</h2>
              <Bars
                rows={ARMS.map((p) => ({
                  label: POLICY_LABEL[p],
                  value: data.summary[p].interventions_per_run,
                  color: POLICY_COLOR[p],
                }))}
                format={(v) => v.toFixed(1)}
              />
              <p className="card-sub">
                HorizonGuard intervenes about as often as the alarm, but chooses the strength and the moment,
                which is where the extra rescues come from.
              </p>
            </div>
          </div>

          <section className="section">
            <h2>Every seed</h2>
            <div className="card">
              <p className="card-sub" style={{ marginTop: 0 }}>
                Filled squares violated. Click a square to replay that run.
              </p>
              <div style={{ overflowX: "auto" }}>
                {ARMS.map((p) => (
                  <div key={p} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
                    <span style={{ width: 140, fontSize: 13.5, flex: "none" }}>{POLICY_LABEL[p]}</span>
                    <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                      {data.rows.map((r) => {
                        const violated = r[`${p}_violated`];
                        const id = `ab-${String(r.seed).padStart(3, "0")}-${p}`;
                        return (
                          <button
                            key={id}
                            title={`seed ${r.seed}: ${violated ? "violated" : "on goal"}`}
                            aria-label={`Open seed ${r.seed} under ${POLICY_LABEL[p]}`}
                            onClick={() => navigate(`/runs/${id}`)}
                            style={{
                              width: 16,
                              height: 16,
                              borderRadius: 4,
                              cursor: "pointer",
                              border: `1.5px solid ${violated ? "#b4492a" : "#a8b49c"}`,
                              background: violated ? "#b4492a" : "transparent",
                              padding: 0,
                            }}
                          />
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
