"""Command line entry point.

Each research stage registers a subcommand here so the Makefile can drive the
whole pipeline with one command per stage.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

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
