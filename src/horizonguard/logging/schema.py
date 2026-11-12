"""Trajectory logging schema v1 and JSONL storage.

One record per agent step. Records carry everything the black-box monitor may
read (action, observation, reasoning text, tool call, virtual time) and the
labels the research pipeline adds (fidelity, state, violation). Labels are
never visible to the monitor at run time.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class StepRecord:
    run_id: str
    env: str
    agent: str
    family: str
    scaffold: str
    seed: int
    t: int
    action: str
    action_kind: str  # goal | side | violate | neutral (simulator ground truth)
    observation: str
    reasoning: str
    tool: str
    clock: float  # virtual seconds since episode start
    intervention: int = 0  # strength applied before this step
    blocked: bool = False  # a gate replaced the proposed action
    fidelity: float | None = None
    violated: bool = False
    template: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # Fields the black-box monitor is allowed to see.
    OBSERVABLE = ("t", "action", "observation", "reasoning", "tool", "clock", "intervention")

    def observable(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.OBSERVABLE}

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRecord:
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {version}")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Trajectory:
    steps: list[StepRecord]

    @property
    def run_id(self) -> str:
        return self.steps[0].run_id if self.steps else ""

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def violation_step(self) -> int | None:
        return next((s.t for s in self.steps if s.violated), None)

    @property
    def violated(self) -> bool:
        return self.violation_step is not None

    def fidelity(self) -> list[float]:
        return [s.fidelity if s.fidelity is not None else float("nan") for s in self.steps]


def _open(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8", newline="\n")


def write_jsonl(records: Iterable[StepRecord], path: str | Path, append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open(path, "a" if append else "w") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[StepRecord]:
    with _open(Path(path), "r") as handle:
        for line in handle:
            if line.strip():
                yield StepRecord.from_dict(json.loads(line))


def group_runs(records: Iterable[StepRecord]) -> list[Trajectory]:
    runs: dict[str, list[StepRecord]] = {}
    for record in records:
        runs.setdefault(record.run_id, []).append(record)
    return [Trajectory(sorted(steps, key=lambda s: s.t)) for steps in runs.values()]
