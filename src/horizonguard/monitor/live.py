"""The streaming monitor: one assessment per agent step.

``LiveMonitor`` bundles the trained components and turns an observable prefix
into what the dashboard shows and what the intervention policy acts on:

* risk of violation within the horizon (multi-state hazard, rollout-supervised);
* drifting belief and onset hazard (the state structure);
* predicted net benefit for every intervention strength;
* the certified decision (net benefit above the disruption-certified threshold,
  abstention head not vetoing, outside a cooldown window);
* a fix-by countdown extrapolated from recent predicted benefit;
* e-process wealth for time-uniform false-alarm control.

Artifacts are trained once from a DriftBench-CF directory and pickled, so the
service starts in under a second after the first build.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from horizonguard.effects.policy import EffectAwarePolicy, fix_by_countdown
from horizonguard.embeddings import PrefixFeaturizer
from horizonguard.eval.leaderboard import load_bench
from horizonguard.guarantees.eprocess import BettingEProcess
from horizonguard.guarantees.policy import CertifiedPolicy
from horizonguard.hazard.model import MultiStateHazard
from horizonguard.logging import StepRecord
from horizonguard.prereg import load_preregistration
from horizonguard.spec import Intervention

ARM_NAMES = [s.name.lower() for s in Intervention.active()]


@dataclass
class _X:
    """Minimal stand-in for a dataset when only features are needed."""

    x: np.ndarray


@dataclass
class MonitorArtifacts:
    hazard: MultiStateHazard
    policy: EffectAwarePolicy
    certificate: dict[str, Any]
    eprocess_m0: float
    trained_on: dict[str, Any]


def build_artifacts(bench_dir: str | Path) -> MonitorArtifacts:
    prereg = load_preregistration()
    bench = load_bench(bench_dir)
    train, cal = bench.part("train"), bench.part("calibration")
    hazard = MultiStateHazard(horizon=prereg.horizon_steps).fit(train, bench.runs)
    certified = CertifiedPolicy(method="crc").fit(train)
    cert = certified.certify(cal)
    healthy = [r for r in bench.part_runs("calibration") if not r.violated]
    from horizonguard.eval.dataset import step_scores

    m0 = BettingEProcess.calibrate(
        step_scores(healthy, hazard.score_steps, 2), alpha=prereg.guarantee.alpha
    ).m0
    return MonitorArtifacts(
        hazard=hazard,
        policy=certified.base,
        certificate={
            "method": cert.method,
            "alpha": cert.alpha,
            "delta": cert.delta,
            "lambda": cert.lam if np.isfinite(cert.lam) else None,
            "n_runs": cert.n_runs,
            "calibration_risk": cert.empirical_risk,
            "vacuous": cert.vacuous,
        },
        eprocess_m0=m0,
        trained_on={
            "bench": str(bench_dir),
            "train_runs": len(bench.split.train),
            "calibration_runs": len(bench.split.calibration),
        },
    )


def load_or_build(bench_dir: str | Path, cache: str | Path | None = None) -> MonitorArtifacts:
    cache = Path(cache) if cache else Path(bench_dir) / "monitor.pkl"
    if cache.exists():
        with cache.open("rb") as handle:
            return pickle.load(handle)  # our own artifact, written below
    artifacts = build_artifacts(bench_dir)
    with cache.open("wb") as handle:
        pickle.dump(artifacts, handle)
    return artifacts


@dataclass
class LiveMonitor:
    artifacts: MonitorArtifacts
    cooldown: int = 5
    featurizer: PrefixFeaturizer = field(default_factory=PrefixFeaturizer)
    _steps: list[StepRecord] = field(default_factory=list)
    _benefit_t: list[int] = field(default_factory=list)
    _benefit: list[float] = field(default_factory=list)
    _scores: list[float] = field(default_factory=list)
    _last_alert: int = -(10**9)

    def __post_init__(self) -> None:
        prereg = load_preregistration()
        self.weight = prereg.disruption_weight
        self.eps = prereg.recoverability_eps
        self.eprocess = BettingEProcess(alpha=prereg.guarantee.alpha, m0=self.artifacts.eprocess_m0)

    def reset(self) -> None:
        self._steps, self._benefit_t, self._benefit, self._scores = [], [], [], []
        self._last_alert = -(10**9)

    @property
    def threshold(self) -> float | None:
        lam = self.artifacts.certificate["lambda"]
        return None if lam is None else max(float(lam), 0.0)

    def assess(self, step: StepRecord) -> dict[str, Any]:
        self._steps.append(step)
        x = self.featurizer(self._steps)[None, :]
        hz = self.artifacts.hazard
        out = hz.hazards(x)
        risk = float(hz.risk(x)[0])
        net = self.artifacts.policy.predicted_net(_X(x))[0]
        r, d = self.artifacts.policy.learner.predict_rescue_disruption(_X(x))
        abstain = float(self.artifacts.policy.abstain_probability(_X(x))[0])
        best = int(np.argmax(net))
        t = step.t
        benefit = float((r - d)[0].max())
        self._benefit_t.append(t)
        self._benefit.append(benefit)
        self._scores.append(risk)
        wealth = float(self.eprocess.wealth(np.array(self._scores))[-1])
        thr = self.threshold
        certified = thr is not None and float(net[best]) >= thr and float(net[best]) > 0
        eligible = abstain < self.artifacts.policy.abstain_threshold
        cooling = t - self._last_alert < self.cooldown
        alert = bool(certified and eligible and not cooling)
        if alert:
            self._last_alert = t
        countdown = fix_by_countdown(self._benefit_t, self._benefit, self.eps)
        return {
            "t": t,
            "risk": risk,
            "p_drifting": float(out.p_drifting[0]),
            "onset_hazard": float(out.from_aligned[0, 1]),
            "net_benefit": {ARM_NAMES[i]: float(v) for i, v in enumerate(net)},
            "rescue": {ARM_NAMES[i]: float(v) for i, v in enumerate(r[0])},
            "disruption": {ARM_NAMES[i]: float(v) for i, v in enumerate(d[0])},
            "best_arm": ARM_NAMES[best],
            "abstain_probability": abstain,
            "certified": bool(certified),
            "alert": alert,
            "intervention": ARM_NAMES[best] if alert else None,
            "fix_by": None if countdown is None else round(float(countdown), 1),
            "evidence": wealth,
            "evidence_alarm": bool(wealth >= 1 / self.eprocess.alpha),
        }

    def assess_all(self, steps: Sequence[StepRecord]) -> list[dict[str, Any]]:
        self.reset()
        return [self.assess(s) for s in steps]
