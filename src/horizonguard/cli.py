"""Command line entry point.

Each research stage registers a subcommand here so the Makefile can drive the
whole pipeline with one command per stage.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from horizonguard import __version__


def _cmd_gap_matrix(args: argparse.Namespace) -> int:
    from horizonguard.literature import load_gap_matrix, render_markdown

    matrix = load_gap_matrix(args.path) if args.path else load_gap_matrix()
    problems = matrix.validate()
    for problem in problems:
        print(f"gap matrix: {problem}", file=sys.stderr)
    if args.write:
        text = render_markdown(args.path) if args.path else render_markdown()
        Path(args.write).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.write}")
    unverified = matrix.unverified()
    print(f"{len(matrix.works)} works, {len(unverified)} still need a full-text read")
    return 1 if problems else 0


def _cmd_scoop_watch(args: argparse.Namespace) -> int:
    from horizonguard.literature import scoop_watch

    log = Path(args.log)
    papers, entry = scoop_watch.run(log)
    if args.dry_run:
        print(entry)
    else:
        scoop_watch.append_entry(log, entry)
        print(f"logged {len(papers)} new papers to {log}")
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    from horizonguard.budget import BudgetPlan, CostModel

    plan = BudgetPlan(
        forked_runs=args.runs,
        grid_prefixes=args.grid,
        strengths=args.strengths,
        continuations_per_arm=args.k,
        horizon=args.horizon,
        early_stop_fraction=args.early_stop,
        cost=CostModel(open_weights_share=args.open_share),
    )
    for key, value in plan.summary().items():
        print(f"{key:>24}: {value:,}" if isinstance(value, int) else f"{key:>24}: {value}")
    if args.stages:
        print()
        for stage in plan.staged(args.stages):
            print(
                f"stage {stage['stage']}: {stage['runs']} runs, "
                f"{stage['agent_steps']:,} steps, ${stage['estimated_dollars']:,}"
            )
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    from horizonguard.eval.power import EffectScenario, power_grid, smallest_design, to_markdown

    results = power_grid(
        EffectScenario(harm_share=args.harm_share),
        ks=tuple(args.ks),
        ns=tuple(args.ns),
        sims=args.sims,
        seed=args.seed,
    )
    table = to_markdown(results)
    best = smallest_design(results, target=args.target)
    summary = (
        f"Cheapest design reaching {args.target:.0%} heterogeneity power: "
        f"K={best.k}, {best.n_prefixes} prefixes"
        if best
        else f"No design in the grid reaches {args.target:.0%} heterogeneity power"
    )
    print(table)
    print(summary)
    if args.write:
        text = f"{table}\n{summary}\n"
        Path(args.write).write_text(text, encoding="utf-8", newline="\n")
    return 0


def _cmd_pilot(args: argparse.Namespace) -> int:
    from horizonguard.eval.pilot import run_pilot, to_markdown

    env_rates, agent_rates, profile = run_pilot(args.runs, seed=args.seed, out=args.out)
    text = (
        to_markdown(env_rates, "environment")
        + "\n"
        + to_markdown(agent_rates, "agent")
        + f"\n{profile['steps']} steps in {profile['seconds']:.1f}s "
        f"({profile['steps_per_second']:.0f} steps/s)\n"
    )
    print(text)
    unusable = [r.group for r in env_rates if not r.usable]
    if unusable:
        print(f"violation base rate outside the usable range for: {', '.join(unusable)}")
    return 1 if unusable else 0


def _cmd_data(args: argparse.Namespace) -> int:
    from horizonguard.bench.generate import BenchConfig, generate
    from horizonguard.repro import load_config

    raw = load_config(args.config)
    bench = dict(raw.get("bench", {}))
    bench.setdefault("seed", raw.get("seed", 20261112))
    if args.replicates is not None:
        bench["replicates"] = args.replicates
    cfg = BenchConfig.from_dict(bench)
    out = args.out or raw.get("output_dir", "data/driftbench-cf")
    manifest = generate(cfg, out)
    print(
        f"DriftBench-CF: {manifest['runs']} runs, {manifest['prefixes']} labelled prefixes, "
        f"{manifest['agent_steps']:,} agent steps -> {out}"
    )
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    import json

    from horizonguard.bench.generate import BenchConfig, load_prefixes
    from horizonguard.bench.quality import arm_noise, gate_passes, replicate_agreement, report

    records = load_prefixes(args.data)
    manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    cfg = BenchConfig.from_dict(manifest["config"])
    agreement = (
        replicate_agreement(records, cfg, n=args.replicates, arm=args.arm)
        if args.replicates
        else None
    )
    text = report(records, agreement)
    print(text)
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8", newline="\n")
    return 0 if gate_passes(arm_noise(records)) else 1


def _cmd_baselines(args: argparse.Namespace) -> int:
    from horizonguard.baselines import FeatureRiskMonitor, MarkovMonitor
    from horizonguard.eval.leaderboard import (
        leaderboard,
        load_bench,
        paradox_check,
        shortcut_check,
        to_markdown,
    )

    bench = load_bench(args.data)
    monitors = [FeatureRiskMonitor(), MarkovMonitor()] if args.quick else None
    rows = leaderboard(bench, monitors)
    sc = shortcut_check(bench, n_boot=100 if args.quick else 1000)
    px = paradox_check(bench)
    lo, hi = sc["mae_gain_ci"]
    lines = [
        "# Baseline leaderboard",
        "",
        to_markdown(rows),
        "## Reality check 1: drift beyond shortcuts",
        "",
        f"- rollout MAE, full features {sc['mae_full']:.3f} vs shortcut-only "
        f"{sc['mae_shortcut']:.3f}; gain {sc['mae_gain']:.3f} (95% CI {lo:.3f} to {hi:.3f})",
        f"- AUROC full {sc['auroc_full']:.3f} vs shortcut {sc['auroc_shortcut']:.3f}; "
        f"within-template {sc['within_template_auroc_full']:.3f} vs "
        f"{sc['within_template_auroc_shortcut']:.3f}",
        f"- verdict: {'PASS' if sc['passes'] else 'FAIL'}",
        "",
        "## Reality check 2: Intervention Paradox (probability alarm, reminder)",
        "",
        "| agent | alarm rate | recovery r | disruption d | net benefit |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in px["agents"]:
        lines.append(
            f"| {row['agent']} | {row['alarm_rate']:.2f} | {row['recovery']:.3f} | "
            f"{row['disruption']:.3f} | {row['net_benefit']:.4f} |"
        )
    lines += [
        "",
        f"- overall net benefit (w = {px['weight']}): {px['overall_net_benefit']:.4f}",
        f"- agents hurt by the alarm: {', '.join(px['agents_hurt']) or 'none'}",
        f"- verdict: {'PASS' if px['disruption_nontrivial'] else 'FAIL'}",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8", newline="\n")
    return 0


def _cmd_hazard(args: argparse.Namespace) -> int:
    from horizonguard.eval.leaderboard import load_bench
    from horizonguard.hazard.comparators import SingleEventClassifier
    from horizonguard.hazard.evaluate import (
        ablations,
        attribution,
        comparators,
        curve_markdown,
        lead_curve,
        lead_gain,
        score_monitor,
    )
    from horizonguard.hazard.model import MultiStateHazard
    from horizonguard.prereg import load_preregistration

    prereg = load_preregistration()
    horizon = prereg.horizon_steps
    bench = load_bench(args.data)
    stride = 3 if args.quick else 2
    hazard = score_monitor(MultiStateHazard(horizon=horizon), bench, stride=stride)
    others = [SingleEventClassifier(horizon=horizon)] if args.quick else comparators(horizon)
    scored = [hazard, *(score_monitor(m, bench, stride=stride) for m in others)]
    budget = prereg.false_alarm_budget_per_1000
    lines = [
        "# Multi-state hazard model",
        "",
        "## Lead-time curve (test + OOD, mean lead in steps, misses count as 0)",
        "",
        curve_markdown(lead_curve(scored)),
        f"## Gain over each comparator at {budget:g} false alarms per 1,000 steps",
        "",
    ]
    gains = []
    for other in scored[1:]:
        point, lo, hi = lead_gain(hazard, other, budget, n_boot=500 if args.quick else 2000)
        gains.append((other.name, point, lo, hi))
        lines.append(f"- vs {other.name}: {point:+.2f} steps (95% CI {lo:+.2f} to {hi:+.2f})")
    best = min(gains, key=lambda g: g[1])  # the strongest comparator gives the smallest gain
    verdict = "SUPPORTED" if best[2] > 0 else "NOT SUPPORTED"
    lines += ["", f"H1 (lead-time gain over the best baseline): {verdict}", ""]
    if not args.quick:
        lines += [
            "## Ablations",
            "",
            curve_markdown(
                lead_curve(
                    [hazard, *(score_monitor(m, bench, stride=stride) for m in ablations(horizon))]
                )
            ),
        ]
    model = MultiStateHazard(horizon=horizon).fit(bench.part("train"), bench.runs)
    attr = attribution(model, bench.data.feature_names)
    lines += ["## Early signature (standardised coefficients)", ""]
    for head, info in attr.items():
        groups = ", ".join(f"{g} {v:.2f}" for g, v in info["by_group"].items())
        top = ", ".join(f"{n} ({c:+.2f})" for n, c in info["top"][:5])
        lines.append(f"- **{head}**: mean |coef| by group: {groups}; top: {top}")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8", newline="\n")
    return 0


def _cmd_effects(args: argparse.Namespace) -> int:
    from horizonguard.effects.evaluate import (
        evaluate_policies,
        harm_analysis,
        learner_table,
        recoverability,
    )
    from horizonguard.eval.leaderboard import load_bench

    bench = load_bench(args.data)
    lines = ["# Recoverability and intervention effects", "", "## Effect learners (test + OOD)", ""]
    lines += ["| learner | corr | slope | binned error | MAE |", "| --- | --- | --- | --- | --- |"]
    for r in learner_table(bench):
        lines.append(
            f"| {r['learner']} | {r['corr']:.3f} | {r['slope']:.3f} | "
            f"{r['binned_error']:.3f} | {r['mae']:.3f} |"
        )
    res = evaluate_policies(bench, args.data)
    lines += [
        "",
        f"## Policy value on fresh forks (net benefit, w = {res['weight']}, "
        f"{res['n_prefixes']} prefixes)",
        "",
        "| policy | value | 95% CI | intervene rate |",
        "| --- | --- | --- | --- |",
    ]
    for r in res["policies"]:
        lo, hi = r["ci"]
        lines.append(
            f"| {r['policy']} | {r['value']:.4f} | {lo:.4f} to {hi:.4f} | "
            f"{r['intervene_rate']:.2f} |"
        )
    c = res["cells"]
    lines += [
        "",
        "## RQ2 gate (agent x environment cells)",
        "",
        f"- cells where the probability alarm is at or below zero: {c['prob_failing_cells']}",
        f"- effect-aware value positive with CI excluding zero: {c['effect_wins']} "
        f"({c['share']:.0%}, target {c['target']:.0%})",
        f"- effect-aware no worse than doing nothing: {c['effect_no_worse']}",
        f"- verdict: {'PASS' if c['passes'] else 'NOT MET'}",
        "",
        "## Recoverability horizon (fix-by)",
        "",
    ]
    for variant, err in recoverability(bench).items():
        lines.append(
            f"- {variant}: recoverable agreement {err['recoverable_agreement']:.2f}, "
            f"mean absolute error {err['mean_abs_error']:.2f} steps, signed "
            f"{err['mean_signed_error']:+.2f}, early or exact {err['share_early_or_exact']:.2f}"
        )
    harm = harm_analysis(bench)
    lines += ["", "## When intervening hurts", ""]
    lines.append(
        "- share of effect variance explained by compliance: "
        + ", ".join(f"{a} {v:.2f}" for a, v in harm["compliance_r2"].items())
    )
    lines.append("- mean disruption by agent (reminder, restatement, gate, escalation):")
    for agent, d in harm["disruption_by_agent"].items():
        lines.append(f"  - {agent}: {', '.join(f'{x:.3f}' for x in d)}")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8", newline="\n")
    return 0


def _cmd_guarantees(args: argparse.Namespace) -> int:
    from horizonguard.eval.leaderboard import load_bench
    from horizonguard.guarantees.evaluate import (
        coverage_study,
        held_out,
        sample_complexity,
        time_uniform,
    )
    from horizonguard.prereg import load_preregistration

    prereg = load_preregistration()
    alpha = prereg.guarantee.alpha
    bench = load_bench(args.data)
    reps = 20 if args.quick else 200
    lines = [
        "# Disruption-controlled guarantees",
        "",
        "## Held-out disruption (certified on calibration runs, measured on fresh forks)",
        "",
        "| method | alpha | lambda | vacuous | held-out risk | rescue | retention | act rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in held_out(bench, args.data):
        lam = "never" if not np.isfinite(r["lambda"]) else f"{r['lambda']:.3f}"
        lines.append(
            f"| {r['method']} | {r['alpha']:.2f} | {lam} | {r['vacuous']} | "
            f"{r['held_risk']:.4f} | {r['held_rescue']:.4f} | {r['retention']:.2f} | "
            f"{r['act_rate']:.2f} |"
        )
    cov = coverage_study(bench, alpha, reps=reps)
    lines += [
        "",
        f"## Coverage over {reps} resampled calibration splits (alpha = {alpha}, "
        f"delta = {cov['delta']})",
        "",
        f"- LTT: held-out risk above alpha in {cov['ltt_violation_rate']:.1%} of resamples "
        f"(must be at most delta); vacuous in {cov['ltt_vacuous_rate']:.0%}",
        f"- CRC: mean held-out risk {cov['crc_mean_risk']:.4f} (must be at most alpha)",
        "",
        "## Sample complexity",
        "",
        "| calibration runs | LTT vacuous | LTT violations | LTT retention | CRC mean risk |",
        "| --- | --- | --- | --- | --- |",
    ]
    sizes = (10, 20) if args.quick else (20, 40, 80, 120)
    for r in sample_complexity(bench, alpha, sizes=sizes, reps=max(10, reps // 3)):
        lines.append(
            f"| {r['n_cal']:.0f} | {r['ltt_vacuous_rate']:.0%} | {r['ltt_violation_rate']:.1%} | "
            f"{r['ltt_retention']:.2f} | {r['crc_mean_risk']:.4f} |"
        )
    tu = time_uniform(bench)
    lines += [
        "",
        "## Time-uniform false alarms (betting e-process vs fixed threshold)",
        "",
        f"- e-process (m0 = {tu['m0']:.3f}): healthy runs with any false alarm "
        f"{tu['eprocess_run_fpr']:.1%} (bound {tu['alpha']:.0%}), detection "
        f"{tu['eprocess_detection']:.2f}, mean lead {tu['eprocess_mean_lead']:.2f} steps",
        f"- fixed threshold at the per-step budget: healthy runs with a false alarm "
        f"{tu['fixed_run_fpr']:.1%}, detection {tu['fixed_detection']:.2f}, mean lead "
        f"{tu['fixed_mean_lead']:.2f} steps",
        f"- held-out healthy runs: {tu['healthy_runs']}",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8", newline="\n")
    return 0


def _cmd_transfer(args: argparse.Namespace) -> int:
    import os

    from horizonguard.baselines import FeatureRiskMonitor, RuleMonitor
    from horizonguard.eval.dataset import step_scores
    from horizonguard.eval.leaderboard import load_bench
    from horizonguard.eval.metrics import threshold_at_budget
    from horizonguard.eval.scaling import scaling_study
    from horizonguard.prereg import load_preregistration
    from horizonguard.transfer.coldstart import cold_start_curve
    from horizonguard.transfer.matrix import summarise, transfer_matrix

    prereg = load_preregistration()
    bench = load_bench(args.data)
    lines = ["# Transfer, cold start, limits and stealth drift", "", "## Transfer matrix", ""]
    lines += [
        "| axis | held out | risk AUROC drop | risk MAE ratio | effect MAE ratio | "
        "guarantee risk (unseen) | holds |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = transfer_matrix(bench.data, alpha=prereg.guarantee.alpha)
    for r in rows:
        lines.append(
            f"| {r['axis']} | {r['held_out']} | {r['risk_auroc_drop']:.3f} | "
            f"{r['risk_mae_ratio']:.2f} | {r['effect_mae_ratio']:.2f} | "
            f"{r['guarantee_risk_ood']:.4f} | {r['guarantee_holds_ood']} |"
        )
    s = summarise(rows, prereg.thresholds["transfer_drop_max"])
    lines += [
        "",
        f"- H4a (signatures transfer, median AUROC drop {s['median_risk_auroc_drop']:.3f} "
        f"<= {prereg.thresholds['transfer_drop_max']}): "
        f"{'SUPPORTED' if s['h4a_supported'] else 'NOT SUPPORTED'}",
        f"- H4b (effects degrade more than risk; median MAE ratio effect "
        f"{s['median_effect_mae_ratio']:.2f} vs risk {s['median_risk_mae_ratio']:.2f}): "
        f"{'SUPPORTED' if s['h4b_supported'] else 'NOT SUPPORTED'}",
        f"- disruption certificate held on {s['share_guarantee_holds']:.0%} of unseen splits",
        "",
        "## Cold start (new family, alpha = 0.02)",
        "",
        "| family | N target runs | source | target | pooled | weighted | effect MAE |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    reps = 5 if args.quick else 30
    families = ("birch",) if args.quick else ("birch", "delta", "cobalt")
    for fam in families:
        for r in cold_start_curve(bench.data, "family", fam, alpha=0.02, reps=reps):
            cells = []
            for m in ("source", "target", "pooled", "weighted"):
                state = (
                    "vacuous"
                    if r[f"{m}_vacuous"] > 0.5
                    else (f"rescue {r[f'{m}_rescue']:.3f}, viol {r[f'{m}_violation']:.0%}")
                )
                cells.append(state)
            lines.append(
                f"| {fam} | {r['n_target']} | " + " | ".join(cells) + f" | {r['effect_mae']:.3f} |"
            )
    monitor = FeatureRiskMonitor().fit(bench.part("train"), bench.runs)
    cal = bench.part_runs("calibration")
    viol = [r.violation_step for r in cal]
    thr = threshold_at_budget(
        step_scores(cal, monitor.score_steps, 2),
        viol,
        prereg.false_alarm_budget_per_1000,
        prereg.horizon_steps,
    )
    cells, law = scaling_study(monitor.score_steps, thr, runs_per_cell=8 if args.quick else 40)
    lines += [
        "",
        "## Lead-time scaling",
        "",
        "| drift strength | context | violation | detection | mean lead |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in cells:
        lines.append(
            f"| {c.strength} | {c.context} | {c.violation_rate:.2f} | {c.detection_rate:.2f} | "
            f"{c.mean_lead:.2f} |"
        )
    pre, exp = law["preregistered"], law["exploratory"]
    lines += [
        "",
        f"- preregistered law (log strength, log context): out-of-sample R^2 "
        f"{pre['r2_loo']:.2f} (H5a needs 0.5): "
        f"{'SUPPORTED' if pre['r2_loo'] >= 0.5 else 'NOT SUPPORTED'}",
        f"- exploratory law with (log strength)^2: out-of-sample R^2 {exp['r2_loo']:.2f}, "
        f"coefficients {', '.join(f'{x:+.2f}' for x in exp['coef'])}",
    ]
    if os.environ.get("HORIZONGUARD_ENABLE_ADVERSARY") == "1":
        from horizonguard.adversary import stress_test

        rule = RuleMonitor().fit(bench.part("train"), bench.runs)
        thr2 = threshold_at_budget(
            step_scores(cal, rule.score_steps, 2),
            viol,
            prereg.false_alarm_budget_per_1000,
            prereg.horizon_steps,
        )
        res = stress_test(
            monitor.score_steps, thr, rule.score_steps, thr2, runs=8 if args.quick else 60
        )
        lines += [
            "",
            "## Stealth-drift stress test (simulation only)",
            "",
            "| setting | violating runs | violation rate | evasion | mean lead |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in res.rows:
            lines.append(
                f"| {r['setting']} | {r['violating_runs']} | {r['violation_rate']:.2f} | "
                f"{r['evasion_rate']:.2f} | {r['mean_lead']:.2f} |"
            )
    else:
        lines += ["", "Stealth-drift study skipped (set HORIZONGUARD_ENABLE_ADVERSARY=1)."]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8", newline="\n")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from horizonguard.monitor.server import serve

    server = serve(args.data, args.host, args.port, args.static)
    print(f"HorizonGuard dashboard on http://{args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="horizonguard",
        description="Recoverability-aware early warning of goal drift.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    gap = sub.add_parser("gap-matrix", help="validate and render the literature gap matrix")
    gap.add_argument("--path", help="gap matrix YAML (defaults to docs/literature)")
    gap.add_argument("--write", help="write the markdown table to this file")
    gap.set_defaults(func=_cmd_gap_matrix)

    scoop = sub.add_parser("scoop-watch", help="run the weekly arXiv scoop-watch queries")
    scoop.add_argument("--log", default="docs/literature/scoop-watch.md")
    scoop.add_argument("--dry-run", action="store_true", help="print the entry, do not log it")
    scoop.set_defaults(func=_cmd_scoop_watch)

    budget = sub.add_parser("budget", help="estimate fork-labelling compute and cost")
    budget.add_argument("--runs", type=int, default=1000, help="F, forked runs")
    budget.add_argument("--grid", type=int, default=5, help="G, fork-grid prefixes per run")
    budget.add_argument("--strengths", type=int, default=2, help="ns, intervention strengths")
    budget.add_argument("--k", type=int, default=6, help="K, continuations per arm")
    budget.add_argument("--horizon", type=int, default=20, help="H, steps per continuation")
    budget.add_argument("--early-stop", type=float, default=0.0, help="share of steps saved")
    budget.add_argument("--open-share", type=float, default=0.7, help="open-weights share")
    budget.add_argument("--stages", type=int, default=0, help="print a staged generation plan")
    budget.set_defaults(func=_cmd_budget)

    pw = sub.add_parser("power", help="simulation-based power analysis for fork effects")
    pw.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6, 8, 12])
    pw.add_argument("--ns", type=int, nargs="+", default=[50, 100, 200, 400])
    pw.add_argument("--sims", type=int, default=200)
    pw.add_argument("--harm-share", type=float, default=0.3)
    pw.add_argument("--target", type=float, default=0.8)
    pw.add_argument("--seed", type=int, default=0)
    pw.add_argument("--write", help="also write the table to this file")
    pw.set_defaults(func=_cmd_power)

    pilot = sub.add_parser("pilot", help="run the Phase 3 pilot and report base rates")
    pilot.add_argument("--runs", type=int, default=50, help="runs per environment")
    pilot.add_argument("--seed", type=int, default=0)
    pilot.add_argument("--out", help="also write trajectories to this JSONL file")
    pilot.set_defaults(func=_cmd_pilot)

    data = sub.add_parser("data", help="generate DriftBench-CF trajectories and fork labels")
    data.add_argument("--config", default="configs/driftbench.yaml")
    data.add_argument("--out", help="output directory (defaults to the config)")
    data.add_argument("--replicates", type=int, help="override runs per cell")
    data.set_defaults(func=_cmd_data)

    audit = sub.add_parser("audit", help="label-noise audit of a generated DriftBench-CF")
    audit.add_argument("--data", default="data/driftbench-cf")
    audit.add_argument("--replicates", type=int, default=40, help="prefixes to re-fork")
    audit.add_argument("--arm", type=int, default=3, help="arm index for replicate checks")
    audit.add_argument("--write", help="also write the report to this file")
    audit.set_defaults(func=_cmd_audit)

    bl = sub.add_parser("baselines", help="baseline leaderboard and reality checks")
    bl.add_argument("--data", default="data/driftbench-cf")
    bl.add_argument("--write", help="also write the report to this file")
    bl.add_argument("--quick", action="store_true", help="two monitors, fewer bootstraps")
    bl.set_defaults(func=_cmd_baselines)

    hz = sub.add_parser("hazard", help="evaluate the multi-state hazard model (Phase 6)")
    hz.add_argument("--data", default="data/driftbench-cf")
    hz.add_argument("--write", help="also write the report to this file")
    hz.add_argument("--quick", action="store_true", help="one comparator, no ablations")
    hz.set_defaults(func=_cmd_hazard)

    ef = sub.add_parser("effects", help="effect models, policy value and recoverability (Phase 7)")
    ef.add_argument("--data", default="data/driftbench-cf")
    ef.add_argument("--write", help="also write the report to this file")
    ef.set_defaults(func=_cmd_effects)

    gu = sub.add_parser("guarantees", help="certified disruption risk and coverage (Phase 8)")
    gu.add_argument("--data", default="data/driftbench-cf")
    gu.add_argument("--write", help="also write the report to this file")
    gu.add_argument("--quick", action="store_true", help="fewer resamples and sizes")
    gu.set_defaults(func=_cmd_guarantees)

    tr = sub.add_parser("transfer", help="transfer, cold start, scaling and stress (Phase 9)")
    tr.add_argument("--data", default="data/driftbench-cf")
    tr.add_argument("--write", help="also write the report to this file")
    tr.add_argument("--quick", action="store_true", help="smaller studies")
    tr.set_defaults(func=_cmd_transfer)

    sv = sub.add_parser("serve", help="run the monitor API and dashboard")
    sv.add_argument("--data", default="data/driftbench-cf")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--static", default="dashboard/dist")
    sv.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
