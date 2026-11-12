"""Compute and label budget for fork-based labelling.

Continuations needed = F x G x (1 + ns) x K, where F is the number of forked
runs, G the number of fork-grid prefixes per run, ns the number of intervention
strengths (the +1 is the no-intervention arm) and K the continuations per arm,
each at most H steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CostModel:
    """Rough per-step costs in US dollars. Replace after the Phase 3 pilot."""

    open_weights_per_step: float = 0.0004
    api_per_step: float = 0.004
    open_weights_share: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 <= self.open_weights_share <= 1.0:
            raise ValueError("open_weights_share must lie in [0, 1]")

    def per_step(self) -> float:
        share = self.open_weights_share
        return share * self.open_weights_per_step + (1 - share) * self.api_per_step


@dataclass(frozen=True)
class BudgetPlan:
    forked_runs: int  # F
    grid_prefixes: int  # G
    strengths: int  # ns
    continuations_per_arm: int  # K
    horizon: int  # H
    early_stop_fraction: float = 0.0  # share of steps saved by terminating decided forks
    cost: CostModel = field(default_factory=CostModel)

    def __post_init__(self) -> None:
        for name in ("forked_runs", "grid_prefixes", "continuations_per_arm", "horizon"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.strengths < 0:
            raise ValueError("strengths must be non-negative")
        if not 0.0 <= self.early_stop_fraction < 1.0:
            raise ValueError("early_stop_fraction must lie in [0, 1)")

    @property
    def arms(self) -> int:
        return 1 + self.strengths

    @property
    def continuations(self) -> int:
        return self.forked_runs * self.grid_prefixes * self.arms * self.continuations_per_arm

    @property
    def agent_steps(self) -> int:
        return round(self.continuations * self.horizon * (1.0 - self.early_stop_fraction))

    @property
    def dollars(self) -> float:
        return self.agent_steps * self.cost.per_step()

    def summary(self) -> dict[str, float | int]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "cost"},
            "arms": self.arms,
            "continuations": self.continuations,
            "agent_steps": self.agent_steps,
            "open_weights_steps": round(self.agent_steps * self.cost.open_weights_share),
            "estimated_dollars": round(self.dollars, 2),
        }

    def staged(self, stages: int = 4) -> list[dict[str, float | int]]:
        """Split generation into stages with a stop rule checked after each one."""
        if stages <= 0:
            raise ValueError("stages must be positive")
        out = []
        done = 0
        for i in range(stages):
            runs = self.forked_runs * (i + 1) // stages - done
            done += runs
            stage = BudgetPlan(
                runs,
                self.grid_prefixes,
                self.strengths,
                self.continuations_per_arm,
                self.horizon,
                self.early_stop_fraction,
                self.cost,
            )
            out.append({"stage": i + 1, "runs": runs, **_money(stage), "cumulative_runs": done})
        return out


def _money(plan: BudgetPlan) -> dict[str, float | int]:
    return {"agent_steps": plan.agent_steps, "estimated_dollars": round(plan.dollars, 2)}


def blueprint_example() -> BudgetPlan:
    """The illustrative sizing from section 7 of the blueprint."""
    return BudgetPlan(
        forked_runs=1000, grid_prefixes=5, strengths=2, continuations_per_arm=6, horizon=20
    )
