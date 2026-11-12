"""Load, validate and render the gap matrix kept in docs/literature."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "docs" / "literature" / "gap_matrix.yaml"
ROW_FIELDS = ("target", "signal", "labels", "guarantee", "transfer", "intervention")
VALID_STATUS = {"abstract", "summary", "full-text", "second-reader"}


@dataclass
class GapMatrix:
    columns: dict[str, list[str]]
    works: list[dict[str, Any]]
    pillars: list[dict[str, Any]]
    problems: list[str] = field(default_factory=list)

    @property
    def work_ids(self) -> set[str]:
        return {w["id"] for w in self.works}

    def validate(self) -> list[str]:
        """Return every problem found; an empty list means the matrix passes the gate."""
        problems: list[str] = []
        seen: set[str] = set()
        for work in self.works:
            wid = work.get("id", "<missing id>")
            if wid in seen:
                problems.append(f"duplicate work id {wid}")
            seen.add(wid)
            for col in ROW_FIELDS:
                value = work.get(col)
                if value not in self.columns.get(col, []):
                    problems.append(f"{wid}: {col}={value!r} is not an allowed value")
            if not work.get("leaves_open"):
                problems.append(f"{wid}: missing 'leaves_open'")
            if work.get("status") not in VALID_STATUS:
                problems.append(f"{wid}: status must be one of {sorted(VALID_STATUS)}")
        for pillar in self.pillars:
            pid = pillar.get("id", "<missing id>")
            closest = pillar.get("closest") or []
            if not closest:
                problems.append(f"{pid}: no closest prior work named")
            for ref in closest:
                if ref not in self.work_ids:
                    problems.append(f"{pid}: closest work {ref!r} is not a row of the matrix")
            if len(str(pillar.get("difference", "")).split()) < 10:
                problems.append(f"{pid}: difference statement is too thin to defend")
        self.problems = problems
        return problems

    def unverified(self) -> list[str]:
        """Works that still need a full-text read before citing."""
        return [w["id"] for w in self.works if w.get("status") in {"abstract", "summary"}]

    def to_markdown(self) -> str:
        header = "| Work | " + " | ".join(c.title() for c in ROW_FIELDS) + " | Leaves open |"
        sep = "| --- " * (len(ROW_FIELDS) + 2) + "|"
        lines = [header, sep]
        for w in self.works:
            cells = [w["id"], *(str(w[c]) for c in ROW_FIELDS), w["leaves_open"]]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "| Pillar | Closest work | Difference | Novelty | Scoop risk |"]
        lines.append("| --- | --- | --- | --- | --- |")
        for p in self.pillars:
            lines.append(
                f"| {p['id']} {p['name']} | {', '.join(p['closest'])} | "
                f"{p['difference']} | {p['novelty']} | {p['scoop_risk']} |"
            )
        return "\n".join(lines) + "\n"


def load_gap_matrix(path: str | Path = DEFAULT_PATH) -> GapMatrix:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return GapMatrix(columns=raw["columns"], works=raw["works"], pillars=raw["pillars"])


def render_markdown(path: str | Path = DEFAULT_PATH) -> str:
    matrix = load_gap_matrix(path)
    intro = (
        "# Gap matrix v1\n\n"
        "Generated from `gap_matrix.yaml` with `horizonguard gap-matrix`. Rows marked "
        "*abstract* or *summary* in the YAML still need a full-text read and a second "
        "reader before citing.\n\n"
    )
    return intro + matrix.to_markdown()
