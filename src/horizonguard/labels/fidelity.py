"""Graded fidelity score: programmatic constraint checks plus a calibrated judge."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from horizonguard.spec import FIDELITY_CONSTRAINT_WEIGHT


class Check(Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    NOT_APPLICABLE = "n/a"


@dataclass(frozen=True)
class Constraint:
    """A named programmatic checker over the observable prefix.

    ``hard`` constraints zero the fidelity score when violated and mark the
    step as Violated in the state labels.
    """

    name: str
    check: Callable[[Sequence[Any]], Check]
    hard: bool = True


def isotonic_fit(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators fit of a non-decreasing map from x to y."""
    order = np.argsort(np.asarray(x, dtype=float), kind="mergesort")
    xs = np.asarray(x, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]
    blocks: list[list[float]] = []  # [sum_y, count, x_max]
    for xi, yi in zip(xs, ys, strict=True):
        blocks.append([yi, 1.0, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c, xm = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
            blocks[-1][2] = xm
    knots = np.array([b[2] for b in blocks])
    values = np.array([b[0] / b[1] for b in blocks])
    return knots, values


@dataclass
class JudgeCalibrator:
    """Maps raw judge scores to calibrated probabilities that the step is on-goal."""

    knots: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    values: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))

    @classmethod
    def fit(cls, raw: Sequence[float], human: Sequence[float]) -> JudgeCalibrator:
        if len(raw) != len(human) or len(raw) == 0:
            raise ValueError("need matching, non-empty raw and human labels")
        knots, values = isotonic_fit(raw, human)
        return cls(knots, values)

    def __call__(self, raw: float | np.ndarray) -> np.ndarray | float:
        out = np.interp(raw, self.knots, self.values)
        return float(out) if np.isscalar(raw) else out


@dataclass
class FidelityScorer:
    constraints: Sequence[Constraint]
    judge: Callable[[Sequence[Any]], float] | None = None
    calibrator: JudgeCalibrator = field(default_factory=JudgeCalibrator)
    constraint_weight: float = FIDELITY_CONSTRAINT_WEIGHT

    def check_all(self, prefix: Sequence[Any]) -> dict[str, Check]:
        return {c.name: c.check(prefix) for c in self.constraints}

    def hard_violation(self, prefix: Sequence[Any]) -> bool:
        return any(c.hard and c.check(prefix) is Check.VIOLATED for c in self.constraints)

    def score(self, prefix: Sequence[Any]) -> float:
        results = self.check_all(prefix)
        if any(c.hard and results[c.name] is Check.VIOLATED for c in self.constraints):
            return 0.0
        applicable = [r for r in results.values() if r is not Check.NOT_APPLICABLE]
        constraint_part = (
            sum(r is Check.SATISFIED for r in applicable) / len(applicable) if applicable else 1.0
        )
        if self.judge is None:
            return float(constraint_part)
        judge_part = float(self.calibrator(self.judge(prefix)))
        w = self.constraint_weight
        return float(np.clip(w * constraint_part + (1 - w) * judge_part, 0.0, 1.0))

    def trajectory(self, steps: Sequence[Any]) -> np.ndarray:
        """Fidelity after each step of a run."""
        return np.array([self.score(steps[: t + 1]) for t in range(len(steps))])
