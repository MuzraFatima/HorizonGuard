"""Phase 8 study: held-out validity, coverage, sample complexity and time-uniform alarms."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from horizonguard.baselines import FeatureRiskMonitor
from horizonguard.effects.evaluate import fresh_labels
from horizonguard.eval.dataset import PrefixDataset, step_scores
from horizonguard.eval.leaderboard import Bench
from horizonguard.eval.metrics import lead_times, threshold_at_budget
from horizonguard.guarantees.eprocess import BettingEProcess, detection, empirical_run_fpr
from horizonguard.guarantees.policy import CertifiedPolicy
from horizonguard.guarantees.risk_control import realised
from horizonguard.prereg import load_preregistration


def _realised(policy: CertifiedPolicy, data: PrefixDataset, lam: float) -> dict[str, float]:
    s = policy.scores(data)
    score = np.where(np.isfinite(s.score), s.score, -1e9)
    return realised(score, s.loss, s.rescue, list(s.run_ids), max(lam, 0.0))


def held_out(
    bench: Bench, bench_dir: str | Path, alphas: Sequence[float] = (0.01, 0.02, 0.05)
) -> list[dict[str, Any]]:
    """Certify on calibration runs, measure on fresh forks of test and OOD runs."""
    delta = load_preregistration().guarantee.delta
    train, cal = bench.part("train"), bench.part("calibration")
    held = fresh_labels(bench_dir, bench.data.select_runs(bench.split.test + bench.split.ood_test))
    base = CertifiedPolicy().fit(train)
    uncertified = _realised(base, held, 0.0)
    rows = []
    for method in ("crc", "ltt"):
        for alpha in alphas:
            pol = CertifiedPolicy(method=method, alpha=alpha, delta=delta, base=base.base)
            cert = pol.certify(cal)
            out = _realised(pol, held, cert.lam)
            rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "lambda": cert.lam,
                    "vacuous": cert.vacuous,
                    "cal_risk": cert.empirical_risk,
                    "held_risk": out["risk"],
                    "held_rescue": out["rescue"],
                    "retention": out["rescue"] / uncertified["rescue"]
                    if uncertified["rescue"] > 0
                    else float("nan"),
                    "act_rate": out["act_rate"],
                }
            )
    rows.append(
        {
            "method": "uncertified",
            "alpha": float("nan"),
            "lambda": 0.0,
            "vacuous": False,
            "cal_risk": float("nan"),
            "held_risk": uncertified["risk"],
            "held_rescue": uncertified["rescue"],
            "retention": 1.0,
            "act_rate": uncertified["act_rate"],
        }
    )
    return rows


def coverage_study(
    bench: Bench,
    alpha: float,
    reps: int = 200,
    cal_fraction: float = 0.5,
    n_cal: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Resample calibration / held-out splits of the non-training runs."""
    delta = load_preregistration().guarantee.delta
    rng = np.random.default_rng(seed)
    base = CertifiedPolicy().fit(bench.part("train")).base
    pool_ids = np.array(sorted(bench.split.calibration + bench.split.test + bench.split.ood_test))
    pool = bench.data.select_runs(list(pool_ids))
    uncertified = _realised(CertifiedPolicy(base=base), pool, 0.0)["rescue"]
    stats: dict[str, list[float]] = {
        k: [] for k in ("ltt_viol", "crc_risk", "ltt_ret", "ltt_vac", "crc_viol")
    }
    for _ in range(reps):
        perm = rng.permutation(pool_ids)
        k = n_cal or int(cal_fraction * len(perm))
        cal_ids, test_ids = list(perm[:k]), list(perm[k:])
        cal, test = pool.select_runs(cal_ids), pool.select_runs(test_ids)
        for method in ("ltt", "crc"):
            pol = CertifiedPolicy(method=method, alpha=alpha, delta=delta, base=base)
            cert = pol.certify(cal)
            out = _realised(pol, test, cert.lam)
            if method == "ltt":
                stats["ltt_viol"].append(out["risk"] > alpha)
                stats["ltt_vac"].append(cert.vacuous)
                stats["ltt_ret"].append(out["rescue"] / uncertified if uncertified else np.nan)
            else:
                stats["crc_risk"].append(out["risk"])
                stats["crc_viol"].append(out["risk"] > alpha)
    return {
        "alpha": alpha,
        "delta": delta,
        "n_cal": float(n_cal or int(cal_fraction * len(pool_ids))),
        "ltt_violation_rate": float(np.mean(stats["ltt_viol"])),
        "ltt_vacuous_rate": float(np.mean(stats["ltt_vac"])),
        "ltt_retention": float(np.nanmean(stats["ltt_ret"])),
        "crc_mean_risk": float(np.mean(stats["crc_risk"])),
        "crc_violation_rate": float(np.mean(stats["crc_viol"])),
    }


def sample_complexity(
    bench: Bench, alpha: float, sizes: Sequence[int] = (10, 20, 40, 80), reps: int = 100
) -> list[dict[str, float]]:
    pool = len(bench.split.calibration + bench.split.test + bench.split.ood_test)
    return [coverage_study(bench, alpha, reps=reps, n_cal=n) for n in sizes if n < pool - 5]


def time_uniform(bench: Bench, stride: int = 2) -> dict[str, Any]:
    """Betting e-process vs a fixed per-step threshold, on held-out runs."""
    prereg = load_preregistration()
    alpha = prereg.guarantee.alpha
    monitor = FeatureRiskMonitor().fit(bench.part("train"), bench.runs)
    cal_runs = bench.part_runs("calibration")
    test_runs = bench.part_runs("test") + bench.part_runs("ood_test")
    cal_scores = step_scores(cal_runs, monitor.score_steps, stride)
    test_scores = step_scores(test_runs, monitor.score_steps, stride)
    healthy_cal = [s for s, r in zip(cal_scores, cal_runs, strict=True) if not r.violated]
    healthy_test = [s for s, r in zip(test_scores, test_runs, strict=True) if not r.violated]
    viol = [r.violation_step for r in test_runs]
    proc = BettingEProcess.calibrate(healthy_cal, alpha=alpha)
    e_det = detection(proc, test_scores, viol)
    thr = threshold_at_budget(
        cal_scores,
        [r.violation_step for r in cal_runs],
        prereg.false_alarm_budget_per_1000,
        prereg.horizon_steps,
    )
    fixed_fpr = float(np.mean([np.any(s >= thr) for s in healthy_test])) if healthy_test else 0.0
    lt = lead_times(test_scores, viol, thr)
    return {
        "alpha": alpha,
        "m0": proc.m0,
        "eprocess_run_fpr": empirical_run_fpr(proc, healthy_test),
        "eprocess_detection": e_det["detection_rate"],
        "eprocess_mean_lead": e_det["mean_lead"],
        "fixed_run_fpr": fixed_fpr,
        "fixed_detection": float((lt > 0).mean()) if lt.size else float("nan"),
        "fixed_mean_lead": float(lt.mean()) if lt.size else float("nan"),
        "healthy_runs": len(healthy_test),
    }
