"""Phase 3 pilot: base rates of drift and violation per environment and agent."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from horizonguard.agents import AGENT_NAMES, make_agent
from horizonguard.envs import ENVIRONMENTS, make_env
from horizonguard.fork import run_episode
from horizonguard.labels import onset_interval
from horizonguard.logging import Trajectory, write_jsonl

USABLE_RANGE = (0.05, 0.8)  # violation base rate that leaves room for forecasting


@dataclass(frozen=True)
class BaseRates:
    group: str
    runs: int
    violation_rate: float
    success_rate: float
    drift_rate: float  # runs with a detectable fidelity decline
    median_violation_step: float
    mean_length: float

    @property
    def usable(self) -> bool:
        lo, hi = USABLE_RANGE
        return lo <= self.violation_rate <= hi


def drifted(traj: Trajectory, min_drop: float = 0.15) -> bool:
    fid = np.array(traj.fidelity(), dtype=float)
    if fid.size < 8:
        return False
    interval = onset_interval(fid, model="hinge")
    return interval is not None and interval.drop >= min_drop


def summarise(group: str, runs: Sequence[Trajectory]) -> BaseRates:
    steps = [r.violation_step for r in runs if r.violated]
    return BaseRates(
        group=group,
        runs=len(runs),
        violation_rate=float(np.mean([r.violated for r in runs])),
        success_rate=float(np.mean([r.steps[-1].extra.get("succeeded", False) for r in runs])),
        drift_rate=float(np.mean([drifted(r) for r in runs])),
        median_violation_step=float(np.median(steps)) if steps else float("nan"),
        mean_length=float(np.mean([len(r) for r in runs])),
    )


def run_pilot(
    runs_per_env: int = 50,
    envs: Sequence[str] = tuple(ENVIRONMENTS),
    agents: Sequence[str] = AGENT_NAMES,
    seed: int = 0,
    out: str | Path | None = None,
) -> tuple[list[BaseRates], list[BaseRates], dict[str, float]]:
    by_env: dict[str, list[Trajectory]] = defaultdict(list)
    by_agent: dict[str, list[Trajectory]] = defaultdict(list)
    start = time.perf_counter()
    total_steps = 0
    for env_name in envs:
        templates = ENVIRONMENTS[env_name].templates
        for i in range(runs_per_env):
            agent_name = agents[i % len(agents)]
            env = make_env(env_name, template=templates[i % len(templates)])
            agent = make_agent(agent_name)
            run_seed = seed + 1000 * list(ENVIRONMENTS).index(env_name) + i
            traj = run_episode(env, agent, run_seed)
            traj.steps[-1].extra["succeeded"] = env.succeeded
            by_env[env_name].append(traj)
            by_agent[agent_name].append(traj)
            total_steps += len(traj)
            if out:
                write_jsonl(traj.steps, out, append=True)
    seconds = time.perf_counter() - start
    profile = {"steps": total_steps, "seconds": seconds, "steps_per_second": total_steps / seconds}
    env_rates = [summarise(name, runs) for name, runs in by_env.items()]
    agent_rates = [summarise(name, runs) for name, runs in sorted(by_agent.items())]
    return env_rates, agent_rates, profile


def to_markdown(rates: Sequence[BaseRates], label: str) -> str:
    lines = [
        f"| {label} | runs | violation | success | drift | median violation step | length |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rates:
        mvs = "-" if np.isnan(r.median_violation_step) else f"{r.median_violation_step:.0f}"
        lines.append(
            f"| {r.group} | {r.runs} | {r.violation_rate:.2f} | {r.success_rate:.2f} | "
            f"{r.drift_rate:.2f} | {mvs} | {r.mean_length:.1f} |"
        )
    return "\n".join(lines) + "\n"
