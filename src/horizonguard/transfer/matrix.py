"""Transfer study (Pillar 4): risk, effect and guarantee reported separately.

For every leave-one-out split along an axis (family, scaffold, environment,
drift type), models are trained on the remaining runs and evaluated twice:

* **in distribution**: on a run-level hold-out of the seen values;
* **transfer**: on the held-out value.

The transfer gap is the difference. The blueprint hypothesis (H4a / H4b):
drift *signatures* transfer with a small risk drop, while *intervention
response* does not, so effect quality degrades more than risk quality. The
guarantee column asks whether a disruption certificate calibrated on seen
agents still holds on the unseen one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from horizonguard.baselines import FeatureRiskMonitor
from horizonguard.bench.splits import Split, leave_one_out
from horizonguard.effects.learners import DecompositionLearner
from horizonguard.eval.dataset import PrefixDataset
from horizonguard.eval.metrics import auroc, rollout_error
from horizonguard.guarantees.policy import CertifiedPolicy
from horizonguard.guarantees.risk_control import realised

AXES = ("family", "scaffold", "env", "drift_type")


def _risk_metrics(model: FeatureRiskMonitor, data: PrefixDataset) -> dict[str, float]:
    p = model.predict(data)
    return {"auroc": auroc(p, data.risk >= 0.5), "mae": rollout_error(p, data.risk)}


def _effect_metrics(model: DecompositionLearner, data: PrefixDataset) -> dict[str, float]:
    pred, label = np.ravel(model.predict_effects(data)), np.ravel(data.arm("benefit"))
    corr = float(np.corrcoef(pred, label)[0, 1]) if pred.std() > 0 and label.std() > 0 else np.nan
    return {"corr": corr, "mae": float(np.mean(np.abs(pred - label)))}


def _guarantee(policy: CertifiedPolicy, data: PrefixDataset) -> dict[str, float]:
    s = policy.scores(data)
    score = np.where(np.isfinite(s.score), s.score, -1e9)
    out = realised(score, s.loss, s.rescue, list(s.run_ids), policy.threshold)
    return {"risk": out["risk"], "act_rate": out["act_rate"]}


def evaluate_split(
    data: PrefixDataset, split: Split, alpha: float = 0.05, seed: int = 0
) -> dict[str, Any]:
    """Train on seen runs, compare in-distribution hold-out against the unseen value."""
    seen = sorted(split.train + split.calibration)
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(seen))
    n_id = max(1, len(perm) // 5)
    id_test, rest = perm[:n_id], perm[n_id:]
    cal_n = max(1, len(rest) // 4)
    cal, train = rest[:cal_n], rest[cal_n:]
    tr, ca = data.select_runs(train), data.select_runs(cal)
    in_dist, unseen = data.select_runs(id_test), data.select_runs(split.ood_test)
    risk = FeatureRiskMonitor().fit(tr, {})
    effect = DecompositionLearner().fit(tr)
    policy = CertifiedPolicy(method="crc", alpha=alpha).fit(tr)
    policy.certify(ca)
    axis, value = next(iter(split.held_out.items()))
    row = {"axis": axis, "held_out": value[0], "n_unseen": len(unseen)}
    for tag, part in (("id", in_dist), ("ood", unseen)):
        for k, v in _risk_metrics(risk, part).items():
            row[f"risk_{k}_{tag}"] = v
        for k, v in _effect_metrics(effect, part).items():
            row[f"effect_{k}_{tag}"] = v
        for k, v in _guarantee(policy, part).items():
            row[f"guarantee_{k}_{tag}"] = v
    row["risk_auroc_drop"] = row["risk_auroc_id"] - row["risk_auroc_ood"]
    row["effect_corr_drop"] = row["effect_corr_id"] - row["effect_corr_ood"]
    row["risk_mae_ratio"] = row["risk_mae_ood"] / max(row["risk_mae_id"], 1e-9)
    row["effect_mae_ratio"] = row["effect_mae_ood"] / max(row["effect_mae_id"], 1e-9)
    row["guarantee_holds_ood"] = bool(row["guarantee_risk_ood"] <= alpha)
    return row


def transfer_matrix(
    data: PrefixDataset, axes: Sequence[str] = AXES, alpha: float = 0.05
) -> list[dict[str, Any]]:
    rows = []
    for axis in axes:
        for split in leave_one_out(data.records, axis):
            if split.ood_test and len(split.train) > 10:
                rows.append(evaluate_split(data, split, alpha))
    return rows


def summarise(rows: Sequence[dict[str, Any]], max_drop: float) -> dict[str, Any]:
    """Test H4a (risk transfers) and H4b (effects degrade more than risk)."""
    drops = np.array([r["risk_auroc_drop"] for r in rows], dtype=float)
    risk_ratio = np.array([r["risk_mae_ratio"] for r in rows], dtype=float)
    effect_ratio = np.array([r["effect_mae_ratio"] for r in rows], dtype=float)
    return {
        "splits": len(rows),
        "share_risk_within_drop": float(np.mean(drops[np.isfinite(drops)] <= max_drop))
        if np.isfinite(drops).any()
        else float("nan"),
        "median_risk_auroc_drop": float(np.nanmedian(drops)),
        "median_risk_mae_ratio": float(np.nanmedian(risk_ratio)),
        "median_effect_mae_ratio": float(np.nanmedian(effect_ratio)),
        "share_effect_degrades_more": float(np.nanmean(effect_ratio > risk_ratio)),
        "share_guarantee_holds": float(np.mean([r["guarantee_holds_ood"] for r in rows])),
        "h4a_supported": bool(np.nanmedian(drops) <= max_drop),
        "h4b_supported": bool(np.nanmedian(effect_ratio) > np.nanmedian(risk_ratio)),
    }
