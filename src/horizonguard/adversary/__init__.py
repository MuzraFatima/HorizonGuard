"""Stealth-drift adversaries for stress-testing monitors (simulation only).

Access-controlled: importing this package requires
``HORIZONGUARD_ENABLE_ADVERSARY=1``. The code drives synthetic agents inside the
simulator against HorizonGuard's own monitors. It is released only after the
responsible-release review (docs/responsible_release.md), and the published
results report metrics, not optimised evasion strategies.
"""

from __future__ import annotations

import os

if os.environ.get("HORIZONGUARD_ENABLE_ADVERSARY") != "1":
    raise ImportError(
        "horizonguard.adversary is access-controlled; set HORIZONGUARD_ENABLE_ADVERSARY=1 "
        "to run the simulation-only stealth-drift study"
    )

from horizonguard.adversary.stealth import StealthAgent, stress_test

__all__ = ["StealthAgent", "stress_test"]
