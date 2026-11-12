"""Typed access to the preregistered constants."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "preregistration.yaml"


@dataclass(frozen=True)
class Guarantee:
    alpha: float
    delta: float
    min_rescue_retention: float


@dataclass(frozen=True)
class Preregistration:
    horizon_steps: int
    disruption_weight: float
    false_alarm_budget_per_1000: float
    recoverability_eps: float
    guarantee: Guarantee
    thresholds: dict[str, float]
    statistics: dict[str, object]

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if not 0 < self.guarantee.alpha < 1 or not 0 < self.guarantee.delta < 1:
            raise ValueError("alpha and delta must lie in (0, 1)")
        if self.disruption_weight < 0:
            raise ValueError("disruption_weight must be non-negative")


@lru_cache(maxsize=4)
def load_preregistration(path: str | Path = DEFAULT_PATH) -> Preregistration:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Preregistration(
        horizon_steps=int(raw["horizon_steps"]),
        disruption_weight=float(raw["disruption_weight"]),
        false_alarm_budget_per_1000=float(raw["false_alarm_budget_per_1000"]),
        recoverability_eps=float(raw["recoverability_eps"]),
        guarantee=Guarantee(**{k: float(v) for k, v in raw["guarantee"].items()}),
        thresholds={k: float(v) for k, v in raw["thresholds"].items()},
        statistics=dict(raw["statistics"]),
    )
