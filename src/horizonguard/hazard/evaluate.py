"""Phase 6 evaluation: lead-time curves, the gain test, ablations and attribution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from horizonguard.baselines import FeatureRiskMonitor, Monitor
from horizonguard.eval.dataset import step_scores
from horizonguard.eval.leaderboard import Bench
from horizonguard.eval.metrics import (
    auroc,
    bootstrap_ci,
    ece,
    lead_times,
    rollout_error,
    threshold_at_budget,
)
from horizonguard.hazard.comparators import LogisticHazard, PCAAnomaly, SingleEventClassifier
from horizonguard.hazard.model import MultiStateHazard
from horizonguard.prereg import load_preregistration

BUDGETS = (1.0, 2.0, 5.0, 10.0, 20.0)


@dataclass
class Scored:
    name: str
    scores: list[np.ndarray]  # per evaluation run
    violations: list[int | None]
    prefix_risk: np.ndarray
    rollout_risk: np.ndarray


def score_monitor(
    monitor: Monitor, bench: Bench, parts: Sequence[str] = ("test", "ood_test"), stride: int = 2
) -> Scored:
    monitor.fit(bench.part("train"), bench.runs)
    ids = [r for p in parts for r in getattr(bench.split, p)]
    runs = [bench.runs[r] for r in ids if r in bench.runs]
    data = bench.data.select_runs(ids)
    return Scored(
        monitor.name,
        step_scores(runs, monitor.score_steps, stride=stride),
        [r.violation_step for r in runs],
        monitor.predict(data),
        data.risk,
    )


def mean_leads(s: Scored, budget: float, horizon: int) -> np.ndarray:
    """Per violating run lead (0 = missed) at a false-alarm budget."""
    thr = threshold_at_budget(s.scores, s.violations, budget, horizon)
    return lead_times(s.scores, s.violations, thr)


def lead_curve(scored: Sequence[Scored], budgets: Sequence[float] = BUDGETS) -> list[dict]:
    horizon = load_preregistration().horizon_steps
    rows = []
    for s in scored:
        row: dict[str, Any] = {"monitor": s.name}
        for b in budgets:
            lt = mean_leads(s, b, horizon)
            row[f"lead@{b:g}"] = float(lt.mean()) if lt.size else float("nan")
            row[f"detect@{b:g}"] = float((lt > 0).mean()) if lt.size else float("nan")
        row["rollout_mae"] = rollout_error(s.prefix_risk, s.rollout_risk)
        row["ece"] = ece(s.prefix_risk, s.rollout_risk)
        row["auroc"] = auroc(s.prefix_risk, s.rollout_risk >= 0.5)
        rows.append(row)
    return rows


def lead_gain(
    a: Scored, b: Scored, budget: float, n_boot: int = 2000
) -> tuple[float, float, float]:
    """Mean lead of ``a`` minus ``b`` at a budget, bootstrap CI over violating runs."""
    horizon = load_preregistration().horizon_steps
    la, lb = mean_leads(a, budget, horizon), mean_leads(b, budget, horizon)
    groups = [np.array([x, y]) for x, y in zip(la, lb, strict=True)]
    return bootstrap_ci(groups, lambda g: float(np.mean([x[0] - x[1] for x in g])), n=n_boot)


def comparators(horizon: int) -> list[Monitor]:
    return [
        FeatureRiskMonitor(name="probability"),
        SingleEventClassifier(horizon=horizon),
        LogisticHazard(horizon=horizon),
        PCAAnomaly(),
    ]


def ablations(horizon: int) -> list[Monitor]:
    return [
        MultiStateHazard(name="hazard: actions only", groups=("act:",), horizon=horizon),
        MultiStateHazard(name="hazard: actions + text", groups=("act:", "emb:"), horizon=horizon),
        MultiStateHazard(
            name="hazard: + geometry (no shortcuts)",
            groups=("act:", "emb:", "geo:"),
            horizon=horizon,
        ),
        MultiStateHazard(
            name="hazard: realised labels only", rollout_supervision=False, horizon=horizon
        ),
    ]


def attribution(model: MultiStateHazard, names: Sequence[str], top: int = 8) -> dict[str, Any]:
    """Which inputs drive the onset and violation hazards (standardised coefficients)."""
    cols = [names[i] for i in model._cols]
    onset = model.head_a.coef_[1:, 1] - model.head_a.coef_[1:, 0]
    violation = model.head_d.coef_[1:, 2] - model.head_d.coef_[1:, 1]
    out: dict[str, Any] = {}
    for label, coef in (("onset", onset), ("violation", violation)):
        groups: dict[str, list[float]] = {}
        for name, c in zip(cols, coef, strict=True):
            groups.setdefault(name.split(":")[0], []).append(abs(float(c)))
        order = np.argsort(-np.abs(coef))[:top]
        out[label] = {
            "by_group": {g: float(np.mean(v)) for g, v in sorted(groups.items())},
            "top": [(cols[i], float(coef[i])) for i in order],
        }
    return out


def curve_markdown(rows: Sequence[dict], budgets: Sequence[float] = BUDGETS) -> str:
    head = ["monitor", *[f"lead@{b:g}" for b in budgets], "detect@5", "MAE", "ECE", "AUROC"]
    lines = ["| " + " | ".join(head) + " |", "| --- " * len(head) + "|"]
    for r in rows:
        cells = [r["monitor"], *[f"{r[f'lead@{b:g}']:.2f}" for b in budgets]]
        cells += [
            f"{r['detect@5']:.2f}",
            f"{r['rollout_mae']:.3f}",
            f"{r['ece']:.3f}",
            f"{r['auroc']:.3f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
