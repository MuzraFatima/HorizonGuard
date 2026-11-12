"""The certified alert policy: effect-aware, with a disruption certificate.

It alerts at a prefix only when three conditions hold together:

1. the predicted net benefit of the best arm is positive;
2. it clears the threshold ``lambda`` certified by risk control on
   fork-labelled calibration runs, so ``R(lambda) <= alpha`` (with probability
   ``1 - delta`` for Learn-then-Test);
3. the abstention head is not confident that every intervention is useless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from horizonguard.effects.policy import EffectAwarePolicy
from horizonguard.eval.dataset import PrefixDataset
from horizonguard.guarantees.risk_control import Certificate, crc_threshold, ltt_threshold
from horizonguard.prereg import load_preregistration


@dataclass
class PolicyScores:
    score: np.ndarray  # best predicted net benefit
    arm: np.ndarray  # best arm, 1..4
    loss: np.ndarray  # fork disruption of that arm
    rescue: np.ndarray  # fork rescue of that arm
    run_ids: np.ndarray


@dataclass
class CertifiedPolicy:
    method: str = "ltt"
    alpha: float | None = None
    delta: float | None = None
    base: EffectAwarePolicy = field(default_factory=EffectAwarePolicy)
    certificate: Certificate | None = None

    def __post_init__(self) -> None:
        prereg = load_preregistration()
        self.alpha = prereg.guarantee.alpha if self.alpha is None else self.alpha
        self.delta = prereg.guarantee.delta if self.delta is None else self.delta

    def fit(self, train: PrefixDataset) -> CertifiedPolicy:
        self.base.fit(train)
        return self

    def scores(self, data: PrefixDataset) -> PolicyScores:
        net = self.base.predicted_net(data)
        arm = net.argmax(axis=1)
        idx = np.arange(len(data))
        eligible = self.base.abstain_probability(data) < self.base.abstain_threshold
        score = np.where(eligible, net.max(axis=1), -np.inf)
        return PolicyScores(
            score=score,
            arm=arm + 1,
            loss=data.arm("disruption")[idx, arm],
            rescue=data.arm("rescue")[idx, arm],
            run_ids=data.meta["run_id"],
        )

    def certify(self, calibration: PrefixDataset) -> Certificate:
        s = self.scores(calibration)
        finite = np.isfinite(s.score)
        score = np.where(finite, s.score, -1e9)
        if self.method == "crc":
            cert = crc_threshold(score, s.loss, list(s.run_ids), self.alpha)
        else:
            cert = ltt_threshold(score, s.loss, list(s.run_ids), self.alpha, self.delta)
        self.certificate = cert
        return cert

    @property
    def threshold(self) -> float:
        if self.certificate is None:
            raise RuntimeError("certify the policy on calibration runs first")
        return max(self.certificate.lam, 0.0)

    def decide(self, data: PrefixDataset) -> np.ndarray:
        s = self.scores(data)
        return np.where(s.score >= self.threshold, s.arm, 0).astype(int)
