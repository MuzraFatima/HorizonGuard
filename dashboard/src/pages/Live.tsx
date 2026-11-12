import { useState } from "react";
import type { Episode, Options, Policy, SimulateParams } from "../api";
import { pretty, simulate } from "../api";
import { EpisodeView } from "../components/EpisodeView";
import { CardSkeleton, ErrorNote, PageHead } from "../components/ui";
import { useApi } from "../hooks";
import { POLICY_LABEL } from "../labels";

const DEFAULTS: SimulateParams = {
  env: "support-desk",
  agent: "fjord-react",
  drift_type: "goal-switch",
  policy: "horizonguard",
  seed: 7,
  max_steps: 50,
};

export default function Live() {
  const options = useApi<Options>("/api/options");
  const [params, setParams] = useState<SimulateParams>(DEFAULTS);
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = <K extends keyof SimulateParams>(key: K, value: SimulateParams[K]) =>
    setParams((p) => ({ ...p, [key]: value }));

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setEpisode(await simulate(params));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const select = (key: "env" | "agent" | "drift_type", label: string, values: string[]) => (
    <label className="field">
      <span>{label}</span>
      <select className="select" value={params[key]} onChange={(e) => set(key, e.target.value)}>
        {values.map((v) => (
          <option key={v} value={v}>
            {pretty(v)}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="page">
      <PageHead
        eyebrow="Live monitor"
        title="Watch a run, step by step"
        lede="Pick an environment, an agent and a source of drift, then let the monitor decide when, and with
        which strength, to re-anchor the agent's goal. Compare against no monitor or a probability-only alarm
        on the same seed."
      />

      <div className="card" style={{ marginBottom: 18 }}>
        {options.error && <ErrorNote message={options.error} />}
        <div className="controls">
          {options.data ? (
            <>
              {select("env", "Environment", options.data.envs)}
              {select("agent", "Agent", options.data.agents)}
              {select("drift_type", "Drift source", options.data.drift_types)}
              <label className="field">
                <span>Policy</span>
                <select
                  className="select"
                  value={params.policy}
                  onChange={(e) => set("policy", e.target.value as Policy)}
                >
                  {options.data.policies.map((p) => (
                    <option key={p} value={p}>
                      {POLICY_LABEL[p]}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field" style={{ minWidth: 90 }}>
                <span>Seed</span>
                <input
                  className="input"
                  type="number"
                  min={0}
                  max={9999}
                  value={params.seed}
                  onChange={(e) => set("seed", Math.max(0, Number(e.target.value) || 0))}
                  style={{ width: 96 }}
                />
              </label>
              <button className="btn" onClick={run} disabled={busy}>
                {busy ? "Running…" : "Run simulation"}
              </button>
            </>
          ) : (
            <div style={{ width: "100%" }}>
              <CardSkeleton lines={1} />
            </div>
          )}
        </div>
        <p className="card-sub" style={{ margin: "12px 0 0" }}>
          Tip: support desk and the fjord family are the fresh, never-trained-on setting used for the A/B
          test.
        </p>
      </div>

      {error && <ErrorNote message={error} />}
      {busy && !episode && <CardSkeleton chart lines={2} />}
      {episode ? (
        <div style={{ opacity: busy ? 0.6 : 1, transition: "opacity .2s" }}>
          <EpisodeView episode={episode} autoplay />
        </div>
      ) : (
        !busy && (
          <div className="card empty">
            Choose settings above and press <strong>Run simulation</strong>.
          </div>
        )
      )}
    </div>
  );
}
