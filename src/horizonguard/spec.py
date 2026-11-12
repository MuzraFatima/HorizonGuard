"""Core types from the specification (docs/spec.md).

Kept dependency-free so every other module can import it.
"""

from __future__ import annotations

from enum import IntEnum

FIDELITY_CONSTRAINT_WEIGHT = 0.5  # w_c in the fidelity score
RECOVERY_RUN = 3  # consecutive steps above onset level that count as recovery
COMPLIANCE_WINDOW = 5  # m, steps after an intervention used to measure compliance


class State(IntEnum):
    ALIGNED = 0
    DRIFTING = 1
    VIOLATED = 2

    @property
    def absorbing(self) -> bool:
        return self is State.VIOLATED


class Transition(IntEnum):
    """Competing transitions of the multi-state goal process."""

    ONSET = 0  # aligned -> drifting
    VIOLATION = 1  # drifting -> violated
    RECOVERY = 2  # drifting -> aligned
    ABRUPT = 3  # aligned -> violated

    @property
    def source(self) -> State:
        return State.ALIGNED if self in (Transition.ONSET, Transition.ABRUPT) else State.DRIFTING

    @property
    def target(self) -> State:
        return {
            Transition.ONSET: State.DRIFTING,
            Transition.VIOLATION: State.VIOLATED,
            Transition.RECOVERY: State.ALIGNED,
            Transition.ABRUPT: State.VIOLATED,
        }[self]

    @classmethod
    def between(cls, source: State, target: State) -> Transition | None:
        for transition in cls:
            if transition.source == source and transition.target == target:
                return transition
        return None


class Intervention(IntEnum):
    """Graded goal re-anchoring family, ordered by strength."""

    NONE = 0
    REMINDER = 1
    RESTATEMENT = 2
    GATE = 3
    ESCALATION = 4

    @classmethod
    def active(cls) -> tuple[Intervention, ...]:
        return tuple(s for s in cls if s is not cls.NONE)


def states_from_labels(onset: int | None, violation: int | None, length: int) -> list[State]:
    """Per-step states for a run with a single drift episode.

    ``onset`` is the first drifting step, ``violation`` the first violated step.
    Either may be None. Steps are 0-indexed.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    states = [State.ALIGNED] * length
    if onset is not None:
        end = violation if violation is not None else length
        for t in range(max(onset, 0), min(end, length)):
            states[t] = State.DRIFTING
    if violation is not None:
        for t in range(max(violation, 0), length):
            states[t] = State.VIOLATED
    return states
