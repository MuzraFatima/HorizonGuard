"""Black-box step encoders and goal-relative geometry.

Only fields in ``StepRecord.OBSERVABLE`` are read. The text encoder is a
signed feature-hashing bag of words: deterministic, dependency-free and good
enough to test whether drift is visible at all. A sentence-embedding model
can replace it behind the same ``encode`` interface.
"""

from __future__ import annotations

import itertools
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from horizonguard.logging.schema import StepRecord

_TOKEN = re.compile(r"[a-z][a-z\-]+")
TOOLS = ("trade", "read", "edit", "shell", "browse", "write", "")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class HashingEncoder:
    dim: int = 64

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for tok in tokens(text):
            h = zlib.crc32(tok.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def step_text(step: StepRecord) -> str:
    return f"{step.action.replace('-', ' ')} {step.reasoning}"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


@dataclass
class PrefixFeaturizer:
    """Fixed-length features of a trajectory prefix.

    Groups (names in ``feature_names``):

    * ``geo:*``   goal-relative geometry against the run's own early anchor;
    * ``act:*``   action and tool statistics over a recent window;
    * ``emb:*``   the exponentially smoothed text embedding;
    * ``short:*`` shortcut channels (step index, clock), kept separate so the
      shortcut controls can drop them.
    """

    encoder: HashingEncoder = field(default_factory=HashingEncoder)
    anchor_steps: int = 3
    window: int = 5
    ema: float = 0.35
    action_buckets: int = 16

    def feature_names(self) -> list[str]:
        geo = ["geo:dist_last", "geo:dist_ema", "geo:angle_ema", "geo:velocity", "geo:accel"]
        act = [f"act:tool_{t or 'none'}" for t in TOOLS] + [
            "act:repeat_rate",
            "act:distinct_rate",
            "act:interventions",
        ]
        act += [f"act:bucket_{i}" for i in range(self.action_buckets)]
        emb = [f"emb:{i}" for i in range(self.encoder.dim)]
        return geo + act + emb + ["short:t", "short:log_t", "short:clock"]

    def __call__(self, steps: Sequence[StepRecord]) -> np.ndarray:
        if not steps:
            return np.zeros(len(self.feature_names()))
        vecs = np.array([self.encoder.encode(step_text(s)) for s in steps])
        anchor = vecs[: self.anchor_steps].mean(axis=0)
        ema = np.zeros(vecs.shape[1])
        dists = []
        for v in vecs:
            ema = self.ema * v + (1 - self.ema) * ema if dists else v
            dists.append(1 - _cos(ema, anchor))
        d = np.array(dists)
        velocity = d[-1] - d[-4] if len(d) >= 4 else 0.0
        prev_velocity = d[-4] - d[-7] if len(d) >= 7 else 0.0
        geo = [
            1 - _cos(vecs[-1], anchor),
            d[-1],
            float(np.arccos(np.clip(1 - d[-1], -1, 1))),
            velocity,
            velocity - prev_velocity,
        ]
        recent = steps[-self.window :]
        n = len(recent)
        tools = [sum(s.tool == t for s in recent) / n for t in TOOLS]
        names = [s.action for s in recent]
        repeat = sum(a == b for a, b in itertools.pairwise(names)) / max(n - 1, 1)
        buckets = np.zeros(self.action_buckets)
        for name in names:
            buckets[zlib.crc32(name.encode()) % self.action_buckets] += 1 / n
        act = [*tools, repeat, len(set(names)) / n, sum(s.intervention > 0 for s in steps)]
        t = len(steps)
        short = [t, np.log1p(t), steps[-1].clock / 3600.0]
        return np.concatenate([geo, act, buckets, ema, short])


@dataclass
class AgentNormalizer:
    """Per-agent z-scoring; unseen agents fall back to pooled statistics."""

    stats: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    pooled: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, x: np.ndarray, agents: Sequence[str]) -> AgentNormalizer:
        agents = np.asarray(agents)
        self.pooled = (x.mean(axis=0), np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0))
        for a in np.unique(agents):
            xa = x[agents == a]
            sd = xa.std(axis=0)
            self.stats[str(a)] = (xa.mean(axis=0), np.where(sd > 1e-9, sd, 1.0))
        return self

    def transform(self, x: np.ndarray, agents: Sequence[str]) -> np.ndarray:
        if self.pooled is None:
            raise RuntimeError("fit the normalizer first")
        out = np.empty_like(x, dtype=float)
        for i, a in enumerate(agents):
            mean, sd = self.stats.get(str(a), self.pooled)
            out[i] = (x[i] - mean) / sd
        return out
