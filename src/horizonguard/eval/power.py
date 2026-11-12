"""Simulation-based power analysis for fork-labelled effects.

Each simulated prefix ``i`` has a baseline violation risk ``p_i`` and a true
rescue and disruption probability. A share ``harm_share`` of prefixes are
"harmful" (intervening mostly disrupts) and the rest are "helpful". With
``K`` seed-paired forks per arm we observe the paired outcome categories,
estimate each prefix's benefit, and run two tests:

* **mean effect:** one-sample t-test that the average benefit is non-zero;
* **heterogeneity:** a Cochran-style Q test that per-prefix benefits differ
  by more than fork noise explains. This is what the effect model needs.

Power is the share of simulated datasets in which a test rejects at ``alpha``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class EffectScenario:
    base_risk: float = 0.5  # mean P(Y(none) = 1)
    helpful_rescue: float = 0.25  # P(rescue) on helpful prefixes
    helpful_disruption: float = 0.02
    harmful_rescue: float = 0.02
    harmful_disruption: float = 0.15
    harm_share: float = 0.3

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Per-prefix category probabilities: rescue, disruption, both fail, both succeed."""
        harmful = rng.random(n) < self.harm_share
        rescue = np.where(harmful, self.harmful_rescue, self.helpful_rescue)
        disrupt = np.where(harmful, self.harmful_disruption, self.helpful_disruption)
        risk = np.clip(rng.beta(4 * self.base_risk + 1, 4 * (1 - self.base_risk) + 1, n), 0, 1)
        rescue = np.minimum(rescue, risk)  # can only rescue runs that would fail
        disrupt = np.minimum(disrupt, 1 - risk)  # can only disrupt runs that would succeed
        both_fail = risk - rescue
        both_ok = 1 - risk - disrupt
        return np.column_stack([rescue, disrupt, both_fail, both_ok])


@dataclass(frozen=True)
class PowerResult:
    k: int
    n_prefixes: int
    mean_effect_power: float
    heterogeneity_power: float
    mean_abs_error: float  # of per-prefix benefit estimates


def _simulate_once(
    scenario: EffectScenario, k: int, n: int, alpha: float, rng: np.random.Generator
) -> tuple[bool, bool, float]:
    probs = scenario.draw(n, rng)
    counts = np.array([rng.multinomial(k, p) for p in probs])
    true_benefit = probs[:, 0] - probs[:, 1]
    est = (counts[:, 0] - counts[:, 1]) / k
    # mean effect
    if np.allclose(est, est[0]):
        mean_reject = bool(abs(est[0]) > 0)
    else:
        mean_reject = bool(stats.ttest_1samp(est, 0.0).pvalue < alpha)
    # heterogeneity: within-prefix variance of the paired difference, pooled
    r, d = counts[:, 0] / k, counts[:, 1] / k
    within = (r + d - (r - d) ** 2) / max(k - 1, 1)  # unbiased per-prefix variance
    pooled_within = max(float(within.mean()), 1e-9)
    q = float(np.sum((est - est.mean()) ** 2) / pooled_within)
    het_reject = bool(stats.chi2.sf(q, df=n - 1) < alpha)
    return mean_reject, het_reject, float(np.mean(np.abs(est - true_benefit)))


def power(
    scenario: EffectScenario,
    k: int,
    n_prefixes: int,
    sims: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> PowerResult:
    if k < 1 or n_prefixes < 2 or sims < 1:
        raise ValueError("need k >= 1, n_prefixes >= 2 and sims >= 1")
    rng = np.random.default_rng(seed)
    results = np.array(
        [_simulate_once(scenario, k, n_prefixes, alpha, rng) for _ in range(sims)], dtype=float
    )
    return PowerResult(
        k=k,
        n_prefixes=n_prefixes,
        mean_effect_power=float(results[:, 0].mean()),
        heterogeneity_power=float(results[:, 1].mean()),
        mean_abs_error=float(results[:, 2].mean()),
    )


def power_grid(
    scenario: EffectScenario,
    ks: tuple[int, ...] = (2, 4, 6, 8, 12),
    ns: tuple[int, ...] = (50, 100, 200, 400),
    sims: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[PowerResult]:
    return [
        power(scenario, k, n, sims, alpha, seed + 31 * i)
        for i, (k, n) in enumerate((k, n) for k in ks for n in ns)
    ]


def smallest_design(results: list[PowerResult], target: float = 0.8) -> PowerResult | None:
    """Cheapest (k x n continuations) design whose heterogeneity power reaches target."""
    ok = [r for r in results if r.heterogeneity_power >= target]
    return min(ok, key=lambda r: (r.k * r.n_prefixes, r.k)) if ok else None


def to_markdown(results: list[PowerResult]) -> str:
    lines = [
        "| K | prefixes | mean-effect power | heterogeneity power | per-prefix MAE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.k} | {r.n_prefixes} | {r.mean_effect_power:.2f} | "
            f"{r.heterogeneity_power:.2f} | {r.mean_abs_error:.3f} |"
        )
    return "\n".join(lines) + "\n"
