"""Wrapper for real LLM agents behind a seeded, cached backend.

The backend is any callable ``(prompt, seed) -> text``. Outputs are cached by
``(sha256(prompt), seed)`` so a fork replays identically even when the backend
is an API. Agents whose provider keeps hidden state, or that ignore the seed,
must set ``counterfactual_ok = False``. They are then excluded from
counterfactual claims (spec, section 7).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from horizonguard.agents.base import Agent, Decision
from horizonguard.envs.base import DriftEnv
from horizonguard.interventions import InterventionMessage

Backend = Callable[[str, int], str]


class CachedBackend:
    def __init__(self, backend: Backend, path: str | Path | None = None):
        self.backend = backend
        self.path = Path(path) if path else None
        self.cache: dict[str, str] = {}
        self.hits = self.misses = 0
        if self.path and self.path.exists():
            self.cache = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(prompt: str, seed: int) -> str:
        return f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}:{seed}"

    def __call__(self, prompt: str, seed: int) -> str:
        key = self.key(prompt, seed)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        out = self.backend(prompt, seed)
        self.cache[key] = out
        return out

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, sort_keys=True), encoding="utf-8")


class LLMAgent(Agent):
    def __init__(
        self,
        backend: Backend,
        name: str = "llm",
        family: str = "llm",
        scaffold: str = "react",
        counterfactual_ok: bool = True,
        memory_steps: int = 12,
    ):
        super().__init__()
        self.backend = backend
        self.name, self.family, self.scaffold = name, family, scaffold
        self.counterfactual_ok = counterfactual_ok
        self.memory_steps = memory_steps

    def build_prompt(self, env: DriftEnv, observation: str) -> str:
        spec = env.goal_spec
        lines = [f"Goal: {spec.goal}" if self.state.goal_in_memory else "Goal: (forgotten)"]
        lines += [f"Constraint: {c}" for c in spec.constraints]
        lines += self.state.memory[-self.memory_steps :]
        lines.append(f"Observation: {observation}")
        lines.append("Available actions: " + ", ".join(a.name for a in env.actions))
        lines.append("Reply with your reasoning, then 'ACTION: <name>'.")
        return "\n".join(lines)

    def act(
        self, env: DriftEnv, observation: str, pressure: float, events: list[str], seed: int
    ) -> Decision:
        if "compaction" in events:
            self.state.goal_in_memory = False
        reply = self.backend(self.build_prompt(env, observation), seed)
        reasoning, _, tail = reply.partition("ACTION:")
        wanted = tail.strip().split()[0] if tail.strip() else ""
        by_name = {a.name: a for a in env.actions}
        action = by_name.get(wanted, env.actions_of("neutral")[0])
        blocked = False
        if self.state.gate_armed and not env.is_allowed(action):
            action, blocked = env.actions_of("neutral")[0], True
        self.state.gate_armed = False
        self.state.memory.append(f"Observation: {observation} | Action: {action.name}")
        return Decision(action, reasoning.strip(), blocked)

    def receive(self, message: InterventionMessage, seed: int, t: int) -> bool:
        if message.is_noop:
            return False
        self.state.memory.append(f"Supervisor: {message.text}")
        self.state.goal_in_memory = True
        if message.blocks_action:
            self.state.gate_armed = True
        return True
