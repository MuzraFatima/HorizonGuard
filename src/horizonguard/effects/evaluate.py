"""Phase 7 evaluation: policy value on fresh forks, the RQ2 gate, heterogeneity."""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from horizonguard.baselines import (
    CIRRouter,
    FeatureRiskMonitor,
    ParadoxRule,
    PeriodicReminder,
    ProbabilityAlarm,
    never,
)
from horizonguard.bench.generate import BenchConfig
from horizonguard.bench.quality import replicate
from horizonguard.effects.learners import (
    LEARNERS,
    DecompositionLearner,
    constant_effect_baseline,
    effect_calibration,
)
from horizonguard.effects.policy import (
    EffectAwarePolicy,
    horizon_errors,
    measured_horizons,
    predicted_horizons,
)
from horizonguard.eval.dataset import PrefixDataset
from horizonguard.eval.leaderboard import Bench
from horizonguard.eval.metrics import bootstrap_ci
from horizonguard.prereg import load_preregistration
from horizonguard.spec import Intervention

FRESH_FILE = "fresh-forks.jsonl.gz"


def fresh_labels(bench_dir: str | Path, data: PrefixDataset) -> PrefixDataset:
    """Re-fork every prefix with disjoint seeds; cached next to the benchmark."""
    path = Path(bench_dir) / FRESH_FILE
    cache: dict[str, dict[str, Any]] = {}
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            cache = {r["id"]: r for r in map(json.loads, handle)}
    manifest = json.loads((Path(bench_dir) / "manifest.json").read_text(encoding="utf-8"))
    cfg = BenchConfig.from_dict(manifest["config"])
    missing = [r for r in data.records if r["id"] not in cache]
    for record in missing:
        cache[record["id"]] = {**record, **replicate(record, cfg)}
    if missing:
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            for rec in cache.values():
                handle.write(json.dumps(rec, sort_keys=True) + "\n")
    return PrefixDataset(
        data.x, data.feature_names, [cache[r["id"]] for r in data.records], data.latent, data.meta
    )


def per_prefix_net(data: PrefixDataset, arms: np.ndarray, weight: float) -> np.ndarray:
    rescue, disrupt = data.arm("rescue"), data.arm("disruption")
    out = np.zeros(len(data))
    idx = np.flatnonzero(arms > 0)
    out[idx] = rescue[idx, arms[idx] - 1] - weight * disrupt[idx, arms[idx] - 1]
    return out


def value_with_ci(
    data: PrefixDataset, arms: np.ndarray, weight: float, n_boot: int = 1000
) -> tuple[float, float, float]:
    net = per_prefix_net(data, arms, weight)
    runs = data.meta["run_id"]
    groups = [net[runs == r] for r in np.unique(runs)]
    return bootstrap_ci(groups, lambda g: float(np.mean(np.concatenate(g))), n=n_boot)


def policy_decisions(bench: Bench, eval_data: PrefixDataset) -> dict[str, np.ndarray]:
    train = bench.part("train")
    risk_model = FeatureRiskMonitor().fit(train, bench.runs)
    risk = risk_model.predict(eval_data)
    return {
        "effect-aware": EffectAwarePolicy().fit(train).decide(eval_data),
        "probability alarm (reminder)": ProbabilityAlarm(0.5).decide(risk, eval_data),
        "probability alarm (escalation)": ProbabilityAlarm(0.5, Intervention.ESCALATION).decide(
            risk, eval_data
        ),
        "CIR-style router": CIRRouter().fit(train).decide(risk, eval_data),
        "Paradox rule": ParadoxRule().fit(train).decide(risk, eval_data),
        "periodic reminders": PeriodicReminder().decide(risk, eval_data),
        "never intervene": never(eval_data),
    }


def evaluate_policies(
    bench: Bench, bench_dir: str | Path, parts: Sequence[str] = ("test", "ood_test")
) -> dict[str, Any]:
    prereg = load_preregistration()
    w = prereg.disruption_weight
    ids = [r for p in parts for r in getattr(bench.split, p)]
    data = bench.data.select_runs(ids)
    fresh = fresh_labels(bench_dir, data)
    decisions = policy_decisions(bench, data)
    table = []
    for name, arms in decisions.items():
        v, lo, hi = value_with_ci(fresh, arms, w)
        table.append(
            {
                "policy": name,
                "value": v,
                "ci": (lo, hi),
                "intervene_rate": float((arms > 0).mean()),
            }
        )
    cells = cell_gate(
        fresh, decisions["effect-aware"], decisions["probability alarm (reminder)"], w
    )
    return {"weight": w, "policies": table, "cells": cells, "n_prefixes": len(data)}


def cell_gate(
    data: PrefixDataset, effect_arms: np.ndarray, prob_arms: np.ndarray, weight: float
) -> dict[str, Any]:
    """RQ2 gate: in cells where the probability alarm is at or below zero, how often
    does the effect-aware policy have a positive value with a CI excluding zero?"""
    agents, envs = data.meta["agent"], data.meta["env"]
    rows = []
    for a in np.unique(agents):
        for e in np.unique(envs):
            m = (agents == a) & (envs == e)
            if m.sum() < 4:
                continue
            sub = data.subset(m)
            ev, elo, ehi = value_with_ci(sub, effect_arms[m], weight, n_boot=500)
            pv, _, _ = value_with_ci(sub, prob_arms[m], weight, n_boot=200)
            rows.append(
                {"agent": str(a), "env": str(e), "effect": ev, "effect_ci": (elo, ehi), "prob": pv}
            )
    failing = [r for r in rows if r["prob"] <= 0]
    wins = [r for r in failing if r["effect_ci"][0] > 0]
    no_harm = [r for r in failing if r["effect_ci"][0] >= -1e-12 and r["effect"] >= 0]
    share = len(wins) / len(failing) if failing else float("nan")
    target = load_preregistration().thresholds["effect_policy_cell_share"]
    return {
        "cells": rows,
        "prob_failing_cells": len(failing),
        "effect_wins": len(wins),
        "effect_no_worse": len(no_harm),
        "share": share,
        "target": target,
        "passes": bool(failing) and share >= target,
    }


def learner_table(bench: Bench) -> list[dict[str, Any]]:
    train = bench.part("train")
    test = bench.data.select_runs(bench.split.test + bench.split.ood_test)
    label = test.arm("benefit")
    rows = []
    for make in LEARNERS:
        model = make().fit(train)
        rows.append(
            {"learner": model.name, **effect_calibration(model.predict_effects(test), label)}
        )
    rows.append(
        {
            "learner": "constant per agent",
            **effect_calibration(constant_effect_baseline(train, test), label),
        }
    )
    return rows


def recoverability(bench: Bench, margin: float = 0.05) -> dict[str, dict[str, float]]:
    """Fix-by accuracy for the point estimate and a conservative one (eps + margin).

    The conservative variant only counts a prefix as still fixable when the
    predicted benefit clears the threshold by ``margin``, so it errs early.
    """
    eps = load_preregistration().recoverability_eps
    test = bench.data.select_runs(bench.split.test + bench.split.ood_test)
    benefit = DecompositionLearner().fit(bench.part("train")).predict_effects(test)
    measured = measured_horizons(test, eps)
    return {
        "point": horizon_errors(predicted_horizons(test, benefit, eps), measured),
        "conservative": horizon_errors(predicted_horizons(test, benefit, eps + margin), measured),
    }


def harm_analysis(bench: Bench, top: int = 6) -> dict[str, Any]:
    """Which prefix features predict disruption, and how much effect compliance explains."""
    train = bench.part("train")
    model = DecompositionLearner().fit(train)
    names = train.feature_names
    out: dict[str, Any] = {"disruption_drivers": {}, "compliance_r2": {}}
    for a, arm in enumerate(("reminder", "restatement", "gate", "escalation")):
        coef = model.disruption[a].coef_[1:]
        order = np.argsort(-np.abs(coef))[:top]
        out["disruption_drivers"][arm] = [(names[i], float(coef[i])) for i in order]
        comp = np.array([r["arms"][a]["compliance"] for r in train.records])
        ben = train.arm("benefit")[:, a]
        ok = np.isfinite(comp)
        r = np.corrcoef(comp[ok], ben[ok])[0, 1] if ok.sum() > 2 and comp[ok].std() > 0 else 0.0
        out["compliance_r2"][arm] = float(r**2)
    by_agent = {}
    d = train.arm("disruption")
    for agent in np.unique(train.meta["agent"]):
        by_agent[str(agent)] = d[train.meta["agent"] == agent].mean(axis=0).round(3).tolist()
    out["disruption_by_agent"] = by_agent
    return out
