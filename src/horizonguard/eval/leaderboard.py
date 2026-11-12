"""Baseline leaderboard and the two Phase 5 reality checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from horizonguard.baselines import (
    FeatureRiskMonitor,
    MarkovMonitor,
    Monitor,
    ProbabilityAlarm,
    default_monitors,
    net_benefit,
    rescue_disruption,
)
from horizonguard.bench.splits import Split, load_splits
from horizonguard.eval.dataset import PrefixDataset, build_prefix_dataset, load_runs, step_scores
from horizonguard.eval.metrics import (
    auroc,
    bootstrap_ci,
    ece,
    healthy_mask,
    lead_time_at_budget,
    lead_times,
    rollout_error,
    threshold_at_budget,
)
from horizonguard.logging import Trajectory
from horizonguard.prereg import load_preregistration


@dataclass
class Bench:
    data: PrefixDataset
    runs: dict[str, Trajectory]
    split: Split

    def part(self, name: str) -> PrefixDataset:
        return self.data.select_runs(getattr(self.split, name))

    def part_runs(self, name: str) -> list[Trajectory]:
        return [self.runs[r] for r in getattr(self.split, name) if r in self.runs]


def load_bench(bench_dir: str | Path, split: str = "default") -> Bench:
    return Bench(
        build_prefix_dataset(bench_dir),
        load_runs(bench_dir),
        load_splits(Path(bench_dir) / "splits.json")[split],
    )


def predict(monitor: Monitor, data: PrefixDataset, runs: Mapping[str, Trajectory]) -> np.ndarray:
    if isinstance(monitor, MarkovMonitor):
        return monitor.predict_runs(data, runs)
    return monitor.predict(data)


def evaluate_monitor(
    monitor: Monitor, bench: Bench, parts: Sequence[str] = ("test", "ood_test"), stride: int = 2
) -> dict[str, Any]:
    prereg = load_preregistration()
    train = bench.part("train")
    monitor.fit(train, bench.runs)
    row: dict[str, Any] = {"monitor": monitor.name, "white_box": monitor.white_box}
    for part in parts:
        data = bench.part(part)
        if len(data) == 0:
            continue
        p = predict(monitor, data, bench.runs)
        row[f"{part}:rollout_mae"] = rollout_error(p, data.risk)
        row[f"{part}:ece"] = ece(p, data.risk)
        row[f"{part}:auroc"] = auroc(p, data.risk >= 0.5)
        runs = bench.part_runs(part)
        scores = step_scores(runs, monitor.score_steps, stride=stride)
        lead = lead_time_at_budget(
            scores,
            [r.violation_step for r in runs],
            prereg.false_alarm_budget_per_1000,
            prereg.horizon_steps,
        )
        row[f"{part}:median_lead"] = lead["median_lead"]
        row[f"{part}:detection"] = lead["detection_rate"]
    return row


def risk_controlled_row(bench: Bench, stride: int = 2) -> dict[str, Any]:
    """Probability scores with the threshold fixed on calibration runs, applied to test."""
    prereg = load_preregistration()
    monitor = FeatureRiskMonitor(name="risk-controlled").fit(bench.part("train"), bench.runs)
    cal = bench.part_runs("calibration")
    cal_scores = step_scores(cal, monitor.score_steps, stride=stride)
    thr = threshold_at_budget(
        cal_scores,
        [r.violation_step for r in cal],
        prereg.false_alarm_budget_per_1000,
        prereg.horizon_steps,
    )
    test = bench.part_runs("test")
    scores = step_scores(test, monitor.score_steps, stride=stride)
    viol = [r.violation_step for r in test]
    lt = lead_times(scores, viol, thr)
    healthy = np.concatenate(
        [
            s[healthy_mask(len(s), v, prereg.horizon_steps)]
            for s, v in zip(scores, viol, strict=True)
        ]
    )
    return {
        "monitor": "risk-controlled",
        "white_box": False,
        "test:median_lead": float(np.median(lt)) if lt.size else float("nan"),
        "test:detection": float((lt > 0).mean()) if lt.size else float("nan"),
        "test:false_alarms_per_1000": float(1000 * (healthy >= thr).mean()),
    }


def leaderboard(bench: Bench, monitors: Sequence[Monitor] | None = None) -> list[dict[str, Any]]:
    prereg = load_preregistration()
    rows = [
        evaluate_monitor(m, bench) for m in (monitors or default_monitors(prereg.horizon_steps))
    ]
    rows.append(risk_controlled_row(bench))
    return rows


# ----- reality check 1: shortcuts ------------------------------------------------------
def shortcut_check(bench: Bench, n_boot: int = 500) -> dict[str, Any]:
    train, test = bench.part("train"), bench.part("test")
    full = FeatureRiskMonitor(name="full", groups=("geo:", "act:", "emb:")).fit(train, bench.runs)
    short = FeatureRiskMonitor(name="short", groups=("short:",)).fit(train, bench.runs)
    pf, ps = full.predict(test), short.predict(test)
    run_ids = test.meta["run_id"]
    groups = [
        np.column_stack([pf[run_ids == r], ps[run_ids == r], test.risk[run_ids == r]])
        for r in np.unique(run_ids)
    ]

    def gain(gs: list[np.ndarray]) -> float:
        g = np.concatenate(gs)
        return float(np.mean(np.abs(g[:, 1] - g[:, 2])) - np.mean(np.abs(g[:, 0] - g[:, 2])))

    point, lo, hi = bootstrap_ci(groups, gain, n=n_boot)
    templates = test.column("template")
    within = [
        auroc(pf[templates == t], test.risk[templates == t] >= 0.5) for t in np.unique(templates)
    ]
    within_short = [
        auroc(ps[templates == t], test.risk[templates == t] >= 0.5) for t in np.unique(templates)
    ]
    return {
        "mae_full": rollout_error(pf, test.risk),
        "mae_shortcut": rollout_error(ps, test.risk),
        "mae_gain": point,
        "mae_gain_ci": (lo, hi),
        "auroc_full": auroc(pf, test.risk >= 0.5),
        "auroc_shortcut": auroc(ps, test.risk >= 0.5),
        "within_template_auroc_full": float(np.nanmean(within)),
        "within_template_auroc_shortcut": float(np.nanmean(within_short)),
        "passes": bool(lo > 0),
    }


# ----- reality check 2: the Intervention Paradox ---------------------------------------
def paradox_check(bench: Bench, threshold: float = 0.5) -> dict[str, Any]:
    prereg = load_preregistration()
    w = prereg.disruption_weight
    monitor = FeatureRiskMonitor().fit(bench.part("train"), bench.runs)
    data = bench.data.select_runs(bench.split.test + bench.split.ood_test)
    risk = monitor.predict(data)
    arms = ProbabilityAlarm(threshold).decide(risk, data)
    rows = []
    agents = data.meta["agent"]
    for a in np.unique(agents):
        m = agents == a
        sub = data.subset(m)
        r, d, rate = rescue_disruption(sub, arms[m])
        rows.append(
            {
                "agent": str(a),
                "alarm_rate": rate,
                "recovery": r,
                "disruption": d,
                "net_benefit": net_benefit(sub, arms[m], w),
            }
        )
    hurt = [row["agent"] for row in rows if row["net_benefit"] < 0]
    max_d = max((row["disruption"] for row in rows), default=0.0)
    return {
        "threshold": threshold,
        "weight": w,
        "agents": rows,
        "agents_hurt": hurt,
        "overall_net_benefit": net_benefit(data, arms, w),
        "disruption_nontrivial": bool(max_d >= 0.05 or hurt),
    }


def to_markdown(rows: Sequence[dict[str, Any]]) -> str:
    cols = [
        ("monitor", "monitor"),
        ("test:rollout_mae", "test MAE"),
        ("test:auroc", "test AUROC"),
        ("test:ece", "test ECE"),
        ("test:median_lead", "test lead"),
        ("test:detection", "test detect"),
        ("ood_test:rollout_mae", "OOD MAE"),
        ("ood_test:auroc", "OOD AUROC"),
        ("ood_test:median_lead", "OOD lead"),
    ]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "| --- " * len(cols) + "|"]
    for row in rows:
        cells = []
        for key, _ in cols:
            v = row.get(key, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        name = cells[0] + (" (white-box)" if row.get("white_box") else "")
        lines.append("| " + " | ".join([name, *cells[1:]]) + " |")
    return "\n".join(lines) + "\n"
