"""Splits for DriftBench-CF.

Rules from the preregistration: split by agent, scaffold, environment and
scenario template, never by step. Every prefix of a run lands in the same
split. Two kinds of split are provided:

* ``default_split``: whole families and drift types held out as an
  out-of-distribution test set; the remaining runs are split into
  train / calibration / in-distribution test by run.
* ``leave_one_out``: one split per value of an axis, for the transfer matrix.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

AXES = ("agent", "family", "scaffold", "env", "template", "drift_type")


@dataclass
class Split:
    name: str
    train: list[str] = field(default_factory=list)  # run ids
    calibration: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    ood_test: list[str] = field(default_factory=list)
    held_out: dict[str, list[str]] = field(default_factory=dict)

    def part_of(self, run_id: str) -> str | None:
        for part in ("train", "calibration", "test", "ood_test"):
            if run_id in getattr(self, part):
                return part
        return None

    def select(self, records: Sequence[dict[str, Any]], part: str) -> list[dict[str, Any]]:
        ids = set(getattr(self, part))
        return [r for r in records if r["run_id"] in ids]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def runs_table(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One row of axis values per run."""
    table: dict[str, dict[str, Any]] = {}
    for r in records:
        table.setdefault(r["run_id"], {axis: r[axis] for axis in AXES})
    return table


def _partition(ids: list[str], fractions: Sequence[float], rng) -> list[list[str]]:
    ids = sorted(ids)
    rng.shuffle(ids)
    cuts = np.cumsum([round(f * len(ids)) for f in fractions[:-1]])
    parts = np.split(np.array(ids, dtype=object), cuts)
    return [sorted(str(x) for x in p) for p in parts]


def default_split(
    records: Sequence[dict[str, Any]],
    held_out_families: Sequence[str] = ("ember",),
    held_out_drift_types: Sequence[str] = ("context",),
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> Split:
    table = runs_table(records)
    ood = [
        rid
        for rid, row in table.items()
        if row["family"] in held_out_families or row["drift_type"] in held_out_drift_types
    ]
    rest = [rid for rid in table if rid not in set(ood)]
    train, cal, test = _partition(rest, fractions, np.random.default_rng(seed))
    return Split(
        name="default",
        train=train,
        calibration=cal,
        test=test,
        ood_test=sorted(ood),
        held_out={"family": list(held_out_families), "drift_type": list(held_out_drift_types)},
    )


def leave_one_out(
    records: Sequence[dict[str, Any]], axis: str, cal_fraction: float = 0.25, seed: int = 0
) -> Iterator[Split]:
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}")
    table = runs_table(records)
    for value in sorted({row[axis] for row in table.values()}):
        test = sorted(rid for rid, row in table.items() if row[axis] == value)
        rest = [rid for rid in table if rid not in set(test)]
        train, cal = _partition(rest, (1 - cal_fraction, cal_fraction), np.random.default_rng(seed))
        yield Split(
            name=f"loo-{axis}-{value}",
            train=train,
            calibration=cal,
            ood_test=test,
            held_out={axis: [value]},
        )


def save_splits(splits: Sequence[Split], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({s.name: s.to_dict() for s in splits}, indent=1), encoding="utf-8"
    )


def load_splits(path: str | Path) -> dict[str, Split]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: Split(**data) for name, data in raw.items()}
