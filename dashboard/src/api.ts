// Typed client for the HorizonGuard API with a small response cache.
// GET responses are kept in memory for `TTL_MS` and identical in-flight requests
// share one network call, so moving between pages does not refetch.

export type ArmName = "reminder" | "restatement" | "gate" | "escalation";

export interface Step {
  t: number;
  action: string;
  reasoning: string;
  events: string[];
  violated: boolean;
  applied: ArmName | null;
  fidelity: number;
  risk: number;
  p_drifting: number;
  onset_hazard: number;
  net_benefit: Record<ArmName, number>;
  best_arm: ArmName;
  abstain_probability: number;
  certified: boolean;
  alert: boolean;
  intervention: ArmName | null;
  fix_by: number | null;
  evidence: number;
  evidence_alarm: boolean;
}

export interface Episode {
  seed: number;
  policy: Policy;
  env: string;
  agent: string;
  violated: boolean;
  succeeded: boolean;
  violation_step: number | null;
  interventions: number;
  timeline: Step[];
}

export type Policy = "none" | "probability" | "horizonguard";

export interface RunSummary {
  id: string;
  env: string;
  agent: string;
  policy: Policy;
  seed: number;
  violated: boolean;
  succeeded: boolean;
  interventions: number;
  length: number;
  peak_risk: number;
}

export interface Page<T> {
  items: T[];
  page: number;
  size: number;
  total: number;
  pages: number;
}

export interface ArmSummary {
  violation_rate: number;
  rescued: number;
  disrupted: number;
  interventions_per_run: number;
}

export interface Certificate {
  method: string;
  alpha: number;
  delta: number | null;
  lambda: number | null;
  n_runs: number;
  calibration_risk: number;
  vacuous: boolean;
}

export interface Finding {
  rq: string;
  claim: string;
  verdict: "supported" | "not supported" | "not met";
  evidence: string;
}

export interface Overview {
  version: string;
  certificate: Certificate;
  ab: Record<Policy, ArmSummary>;
  trained_on: { train_runs: number; calibration_runs: number };
  findings: Finding[];
}

export interface Options {
  envs: string[];
  agents: string[];
  drift_types: string[];
  policies: Policy[];
}

export interface ABRow {
  seed: number;
  none_violated: boolean;
  probability_violated: boolean;
  horizonguard_violated: boolean;
  none_interventions: number;
  probability_interventions: number;
  horizonguard_interventions: number;
}

export interface Guarantees {
  certificate: Certificate;
  eprocess_m0: number;
  studies: {
    sample_complexity: { n: number; vacuous: number; violations: number; retention: number }[];
    coverage: { resamples: number; ltt_violation_rate: number; crc_mean_risk: number };
    time_uniform: {
      eprocess_fpr: number;
      eprocess_detection: number;
      eprocess_lead: number;
      fixed_detection: number;
      fixed_lead: number;
    };
  };
}

const TTL_MS = 60_000;
const cache = new Map<string, { at: number; data: unknown }>();
const inflight = new Map<string, Promise<unknown>>();

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.error) message = body.error;
    } catch {
      /* keep the status text */
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

export function getCached<T>(path: string): T | undefined {
  const hit = cache.get(path);
  return hit && Date.now() - hit.at < TTL_MS ? (hit.data as T) : undefined;
}

export function get<T>(path: string): Promise<T> {
  const cached = getCached<T>(path);
  if (cached !== undefined) return Promise.resolve(cached);
  const pending = inflight.get(path);
  if (pending) return pending as Promise<T>;
  const p = request<T>(path)
    .then((data) => {
      cache.set(path, { at: Date.now(), data });
      return data;
    })
    .finally(() => inflight.delete(path));
  inflight.set(path, p);
  return p;
}

export interface SimulateParams {
  env: string;
  agent: string;
  drift_type: string;
  policy: Policy;
  seed: number;
  max_steps: number;
}

export function simulate(params: SimulateParams): Promise<Episode> {
  const key = `sim:${JSON.stringify(params)}`;
  const cached = getCached<Episode>(key);
  if (cached) return Promise.resolve(cached);
  return request<Episode>("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  }).then((ep) => {
    cache.set(key, { at: Date.now(), data: ep });
    return ep;
  });
}

export const pretty = (s: string) => s.replace(/-/g, " ");
export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;
