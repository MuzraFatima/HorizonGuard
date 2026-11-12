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
