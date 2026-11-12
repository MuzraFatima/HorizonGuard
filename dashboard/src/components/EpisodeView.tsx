import { memo, useEffect, useMemo, useState } from "react";
import type { ArmName, Episode, Step } from "../api";
import { pct, pretty } from "../api";
import { TimelineChart } from "./charts";
import { IconPause, IconPlay } from "./Icons";
import { Chip } from "./ui";

const ARMS: ArmName[] = ["reminder", "restatement", "gate", "escalation"];

const NetBenefit = memo(function NetBenefit({ step }: { step: Step }) {
  const span = Math.max(0.15, ...ARMS.map((a) => Math.abs(step.net_benefit[a])));
  return (
    <div>
      {ARMS.map((arm) => {
        const v = step.net_benefit[arm];
        const width = (Math.abs(v) / span) * 50;
        return (
          <div className="arm-row" key={arm}>
            <span style={{ fontWeight: arm === step.best_arm ? 600 : 400 }}>{arm}</span>
            <div className="arm-track">
              <div className="arm-zero" />
              <div
                className="arm-bar"
                style={{
                  left: v >= 0 ? "50%" : `${50 - width}%`,
                  width: `${width}%`,
                  background: v >= 0 ? "#4f6f41" : "#b4492a",
                }}
              />
            </div>
            <span className="mono" style={{ textAlign: "right" }}>
              {v >= 0 ? "+" : ""}
              {v.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
});

function FixBy({ step }: { step: Step }) {
  if (step.fix_by === null)
    return <div className="stat-value tone-good" style={{ fontSize: 24 }}>no deadline</div>;
  if (step.fix_by <= 0)
    return <div className="stat-value tone-bad" style={{ fontSize: 24 }}>too late</div>;
  return (
    <div className="stat-value tone-warn" style={{ fontSize: 24 }}>
      {step.fix_by.toFixed(1)} <span style={{ fontSize: 14 }}>steps</span>
    </div>
  );
}

export function EpisodeView({ episode, autoplay = false }: { episode: Episode; autoplay?: boolean }) {
  const steps = episode.timeline;
  const [cursor, setCursor] = useState(autoplay ? 0 : steps.length - 1);
  const [playing, setPlaying] = useState(autoplay);

  useEffect(() => {
    setCursor(autoplay ? 0 : steps.length - 1);
    setPlaying(autoplay);
  }, [episode, autoplay, steps.length]);

  useEffect(() => {
    if (!playing) return;
    if (cursor >= steps.length - 1) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setCursor((c) => c + 1), 220);
    return () => window.clearTimeout(id);
  }, [playing, cursor, steps.length]);

  const step = steps[Math.min(cursor, steps.length - 1)];
  const visibleLog = useMemo(() => steps.slice(0, cursor + 1).reverse(), [steps, cursor]);
  if (!step) return <div className="empty">This run has no steps.</div>;

  return (
    <div className="grid" style={{ gap: 18 }}>
      <div className="card">
        <div className="card-head">
          <div>
            <h2>
              {pretty(episode.agent)} in {pretty(episode.env)}
            </h2>
            <div className="card-sub">
              seed {episode.seed} · {episode.interventions} intervention
              {episode.interventions === 1 ? "" : "s"} ·{" "}
              {episode.violated ? `violated at step ${episode.violation_step}` : "no violation"}
            </div>
          </div>
          {episode.violated ? <Chip tone="bad">violated</Chip> : <Chip tone="good">stayed on goal</Chip>}
        </div>
        <TimelineChart steps={steps} cursor={cursor} violationStep={episode.violation_step} />
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 10 }}>
          <button
            className="btn btn-ghost"
            onClick={() => {
              if (cursor >= steps.length - 1) setCursor(0);
              setPlaying((p) => !p);
            }}
            aria-label={playing ? "Pause playback" : "Play the run"}
          >
            {playing ? <IconPause /> : <IconPlay />}
            {playing ? "Pause" : "Replay"}
          </button>
          <input
            type="range"
            min={0}
            max={steps.length - 1}
            value={cursor}
            onChange={(e) => {
              setPlaying(false);
              setCursor(Number(e.target.value));
            }}
            style={{ flex: 1, accentColor: "#b97d12" }}
            aria-label="Step"
          />
          <span className="mono" style={{ minWidth: 70, textAlign: "right" }}>
            step {step.t}
          </span>
        </div>
        <div className="legend" style={{ marginTop: 10 }}>
          <span><i style={{ background: "#b4492a" }} />risk of violation within 20 steps</span>
          <span><i style={{ background: "#4f6f41" }} />goal fidelity</span>
          <span><i style={{ background: "#ecdde7", height: 9 }} />belief the agent is drifting</span>
          <span><i style={{ background: "#e0a83b" }} />intervention sent</span>
        </div>
      </div>

      <div className="grid grid-4">
        <div className="card">
          <div className="stat-label">Risk now</div>
          <div className={`stat-value ${step.risk >= 0.5 ? "tone-bad" : ""}`}>{pct(step.risk)}</div>
          <div className="stat-note">drifting belief {pct(step.p_drifting)}</div>
        </div>
        <div className="card">
          <div className="stat-label">Fix-by countdown</div>
          <FixBy step={step} />
          <div className="stat-note">steps before re-anchoring stops helping</div>
        </div>
        <div className="card">
          <div className="stat-label">Decision</div>
          <div className="stat-value" style={{ fontSize: 22 }}>
            {step.alert ? (
              <span className="tone-warn">send {step.intervention}</span>
            ) : step.certified ? (
              "cooling down"
            ) : (
              "keep watching"
            )}
          </div>
          <div className="stat-note">
            {step.certified ? <Chip tone="good">certified</Chip> : <Chip>below threshold</Chip>}
          </div>
        </div>
        <div className="card">
          <div className="stat-label">Evidence (e-process)</div>
          <div className={`stat-value ${step.evidence_alarm ? "tone-bad" : ""}`}>
            {step.evidence >= 100 ? step.evidence.toExponential(1) : step.evidence.toFixed(2)}
          </div>
          <div className="stat-note">alarm at 20 · false alarms ≤ 5% per run</div>
        </div>
      </div>

      <div className="split">
        <div className="card">
          <div className="card-head">
            <h3>Step log</h3>
            <span className="card-sub">newest first</span>
          </div>
          <div className="step-log">
            <ol>
              {visibleLog.map((s) => (
                <li key={s.t} className={s.t === step.t ? "now" : undefined}>
                  <span className="t">{String(s.t).padStart(2, "0")}</span>
                  <div>
                    <strong style={{ fontWeight: 500 }}>{pretty(s.action)}</strong>
                    {s.applied && (
                      <>
                        {" "}
                        <Chip tone="warn">{s.applied} received</Chip>
                      </>
                    )}
                    {s.violated && (
                      <>
                        {" "}
                        <Chip tone="bad">violation</Chip>
                      </>
                    )}
                    {s.events.length > 0 && (
                      <>
                        {" "}
                        <Chip tone="plum">{s.events.join(", ")}</Chip>
                      </>
                    )}
                    <div className="card-sub">{s.reasoning}</div>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </div>
        <div className="card">
          <div className="card-head">
            <h3>Predicted net benefit</h3>
            <span className="card-sub">rescue − 2 × disruption</span>
          </div>
          <NetBenefit step={step} />
          <p className="card-sub" style={{ marginTop: 12 }}>
            HorizonGuard acts only when the best bar clears the certified threshold and the abstention head
            does not veto. Red bars mean intervening would more likely break a run that was going to succeed.
          </p>
        </div>
      </div>
    </div>
  );
}
