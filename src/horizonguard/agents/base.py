"""Agent interface shared by synthetic, scripted and LLM agents."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from horizonguard.envs.base import Action, DriftEnv
from horizonguard.interventions import InterventionMessage


@dataclass
class AgentState:
    drift: float = 0.0  # latent, simulator-only; never logged as observable
    goal_in_memory: bool = True
    memory: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)  # action kinds queued by planners
    confused: int = 0  # steps of post-disruption confusion left
    gate_armed: bool = False  # the next action passes a constraint gate
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    action: Action
    reasoning: str
    blocked: bool = False


class Agent(ABC):
    name: str = "agent"
    family: str = "none"
    scaffold: str = "none"
    counterfactual_ok: bool = True  # False for agents whose state cannot be snapshotted

    def __init__(self) -> None:
        self.state = AgentState()

    def reset(self) -> None:
        self.state = AgentState()

    def snapshot(self) -> AgentState:
        return copy.deepcopy(self.state)

    def restore(self, snap: AgentState) -> None:
        self.state = copy.deepcopy(snap)

    @abstractmethod
    def act(
        self, env: DriftEnv, observation: str, pressure: float, events: list[str], seed: int
    ) -> Decision: ...

    @abstractmethod
    def receive(self, message: InterventionMessage, seed: int, t: int) -> bool:
        """Apply an intervention; return whether the agent complied."""
