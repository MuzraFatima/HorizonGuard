"""Agent wrappers: synthetic families, scripted control and LLM backends."""

from horizonguard.agents.base import Agent, AgentState, Decision
from horizonguard.agents.llm import CachedBackend, LLMAgent
from horizonguard.agents.synthetic import FAMILIES, FamilyProfile, ScriptedAgent, SyntheticAgent

SCAFFOLDS = ("react", "plan")
AGENT_NAMES = (*(f"{f}-{s}" for f in FAMILIES for s in SCAFFOLDS), "scripted")


def make_agent(name: str) -> Agent:
    """Build an agent from its registry name, e.g. ``"birch-plan"`` or ``"scripted"``."""
    if name == "scripted":
        return ScriptedAgent()
    family, _, scaffold = name.partition("-")
    return SyntheticAgent(family, scaffold or "react")


__all__ = [
    "AGENT_NAMES",
    "FAMILIES",
    "SCAFFOLDS",
    "Agent",
    "AgentState",
    "CachedBackend",
    "Decision",
    "FamilyProfile",
    "LLMAgent",
    "ScriptedAgent",
    "SyntheticAgent",
    "make_agent",
]
