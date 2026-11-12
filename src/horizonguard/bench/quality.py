"""Label-quality audit for DriftBench-CF.

The Phase 4 gate asks whether label noise is small enough for effect
heterogeneity to be detectable. Three checks:

1. **Replicate agreement.** Re-fork a sample of prefixes with disjoint seeds
   and compare the two label sets: correlation, and noise variance estimated
   as half the variance of replicate differences.
2. **Predicted vs observed noise.** The fork standard errors should match the
   replicate noise; if not, labels are noisier than their intervals claim.
3. **Heterogeneity.** A Cochran-style Q test of per-prefix benefit against
   pooled fork noise, per arm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from horizonguard.bench.generate import BenchConfig, RunSpec, _fresh, label_prefix
from horizonguard.fork import ForkEngine, run_episode
from horizonguard.spec import Intervention

REPLICATE_OFFSET = 10_000  # seed index offset, far from any top-up range


@dataclass(frozen=True)
class ArmNoise:
    arm: str
    mean_benefit: float
    mean_se: float
    share_ci_excludes_zero: float
    heterogeneity_q: float
    heterogeneity_p: float
    signal_sd: float  # sqrt(max(var(benefit) - mean(se^2), 0))


@dataclass(frozen=True)
class ReplicateAgreement:
    n: int
    risk_corr: float
    benefit_corr: float
    risk_noise_sd: float
    benefit_noise_sd: float
    predicted_benefit_se: float


def within_variance(record: dict[str, Any], arm_index: int) -> float:
    """Unbiased fork-noise variance of one prefix's benefit estimate.

    Uses the multinomial form ``(r + d - (r - d)^2) / (k - 1)`` so prefixes
    whose forks all agree still contribute an honest, non-zero noise level
    once averaged, unlike the per-prefix paired standard error.
    """
    arm = record["arms"][arm_index]
    r, d, k = arm["rescue"], arm["disruption"], record["k"]
    return (r + d - (r - d) ** 2) / max(k - 1, 1)


def arm_noise(records: Sequence[dict[str, Any]]) -> list[ArmNoise]:
    out = []
    for i, name in enumerate(a["name"] for a in records[0]["arms"]):
        b = np.array([r["arms"][i]["benefit"] for r in records])
        within = np.array([within_variance(r, i) for r in records])
        pooled = float(within.mean())
        q = float(np.sum((b - b.mean()) ** 2) / pooled) if pooled > 0 else 0.0
        se = np.sqrt(pooled)
        out.append(
            ArmNoise(
                arm=name,
                mean_benefit=float(b.mean()),
                mean_se=float(se),
                share_ci_excludes_zero=float(np.mean(np.abs(b) > 1.96 * se)),
                heterogeneity_q=q,
                heterogeneity_p=float(stats.chi2.sf(q, df=len(b) - 1)),
                signal_sd=float(np.sqrt(max(np.var(b, ddof=1) - pooled, 0.0))),
            )
        )
    return out


def replicate(record: dict[str, Any], cfg: BenchConfig) -> dict[str, Any]:
    spec = RunSpec(
        record["env"], record["agent"], record["drift_type"], 0, record["template"], record["seed"]
    )
    env, agent = _fresh(spec, cfg)
    run_episode(env, agent, spec.seed, run_id=record["run_id"], max_steps=record["t"])
    engine = ForkEngine(record["horizon"])
    snap = engine.snapshot(env, agent, record["run_id"], spec.seed)
    result = engine.fork(
        env, agent, snap, tuple(Intervention), k=record["k"], seed_offset=REPLICATE_OFFSET
    )
    return label_prefix(result, record["fidelity_t"])


def replicate_agreement(
    records: Sequence[dict[str, Any]], cfg: BenchConfig, n: int = 40, arm: int = 1, seed: int = 0
) -> ReplicateAgreement:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(n, len(records)), replace=False)
    sample = [records[int(i)] for i in idx]
    reps = [replicate(r, cfg) for r in sample]
    r1 = np.array([r["risk"] for r in sample])
    r2 = np.array([r["risk"] for r in reps])
    b1 = np.array([r["arms"][arm]["benefit"] for r in sample])
    b2 = np.array([r["arms"][arm]["benefit"] for r in reps])
    pooled = float(np.mean([within_variance(r, arm) for r in sample]))

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")

    return ReplicateAgreement(
        n=len(sample),
        risk_corr=corr(r1, r2),
        benefit_corr=corr(b1, b2),
        risk_noise_sd=float(np.sqrt(np.var(r1 - r2, ddof=1) / 2)),
        benefit_noise_sd=float(np.sqrt(np.var(b1 - b2, ddof=1) / 2)),
        predicted_benefit_se=float(np.sqrt(pooled)),
    )


def gate_passes(noise: Sequence[ArmNoise], alpha: float = 0.05) -> bool:
    """Heterogeneity must be detectable for at least one active arm."""
    return any(a.heterogeneity_p < alpha for a in noise)


def report(records: Sequence[dict[str, Any]], agreement: ReplicateAgreement | None = None) -> str:
    noise = arm_noise(records)
    lines = [
        "# DriftBench-CF label-noise report",
        "",
        f"Prefixes: {len(records)}. Forks per arm: {records[0]['k']}.",
        "",
        "| arm | mean benefit | mean SE | CI excludes 0 | signal SD | Q | p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for a in noise:
        lines.append(
            f"| {a.arm} | {a.mean_benefit:.3f} | {a.mean_se:.3f} | "
            f"{a.share_ci_excludes_zero:.2f} | {a.signal_sd:.3f} | {a.heterogeneity_q:.0f} | "
            f"{a.heterogeneity_p:.2g} |"
        )
    if agreement:
        lines += [
            "",
            f"Replicate agreement on {agreement.n} re-forked prefixes (disjoint seeds):",
            "",
            f"- risk correlation {agreement.risk_corr:.2f}, noise SD {agreement.risk_noise_sd:.3f}",
            f"- benefit correlation {agreement.benefit_corr:.2f}, noise SD "
            f"{agreement.benefit_noise_sd:.3f} (predicted SE {agreement.predicted_benefit_se:.3f})",
        ]
    lines += ["", f"Gate (heterogeneity detectable): {'PASS' if gate_passes(noise) else 'FAIL'}"]
    return "\n".join(lines) + "\n"
