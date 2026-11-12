"""Krippendorff's alpha for annotation agreement, with missing values."""

from __future__ import annotations

import numpy as np


def krippendorff_alpha(ratings: np.ndarray, level: str = "interval") -> float:
    """Alpha for a raters x units matrix; NaN marks a missing rating.

    ``level`` is ``"nominal"`` or ``"interval"``. Units with fewer than two
    ratings are ignored, as in the standard definition.
    """
    data = np.asarray(ratings, dtype=float)
    if data.ndim != 2:
        raise ValueError("ratings must be a raters x units matrix")
    if level not in {"nominal", "interval"}:
        raise ValueError("level must be 'nominal' or 'interval'")

    def delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if level == "nominal":
            return (a != b).astype(float)
        return (a - b) ** 2

    units = [col[~np.isnan(col)] for col in data.T]
    units = [u for u in units if u.size >= 2]
    if not units:
        raise ValueError("need at least one unit with two or more ratings")
    n_total = sum(u.size for u in units)

    observed = 0.0
    for u in units:
        d = delta(u[:, None], u[None, :])
        observed += d.sum() / (u.size - 1)
    observed /= n_total

    pooled = np.concatenate(units)
    d_all = delta(pooled[:, None], pooled[None, :])
    expected = d_all.sum() / (n_total * (n_total - 1))
    if expected == 0:
        return 1.0
    return float(1.0 - observed / expected)
