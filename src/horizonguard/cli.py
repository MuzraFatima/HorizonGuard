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
