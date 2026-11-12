# Phase 3 pilot

50 runs per environment, cycling through all 11 agents and every scenario
template. Reproduce with `horizonguard pilot --runs 50`.

## Environments

| environment | runs | violation | success | drift | median violation step | length |
| --- | --- | --- | --- | --- | --- | --- |
| portfolio | 50 | 0.20 | 0.80 | 0.60 | 24 | 23.4 |
| codebase | 50 | 0.40 | 0.60 | 0.72 | 27 | 28.3 |
| research | 50 | 0.46 | 0.54 | 0.78 | 22 | 24.9 |
| sysadmin | 50 | 0.46 | 0.54 | 0.76 | 28 | 27.8 |

All four violation base rates fall in the usable range (0.05 to 0.8), so the
gate is met. Drift (a fitted fidelity decline of at least 0.15) is more common
than violation, which is the regime where early warning and rescue are
possible at all.

## Agents

| agent | runs | violation | success | drift | median violation step | length |
| --- | --- | --- | --- | --- | --- | --- |
| atlas-plan | 20 | 0.05 | 0.95 | 0.70 | 41 | 27.3 |
| atlas-react | 20 | 0.20 | 0.80 | 0.55 | 28 | 25.1 |
| birch-plan | 20 | 0.75 | 0.25 | 0.95 | 24 | 26.3 |
| birch-react | 20 | 0.75 | 0.25 | 1.00 | 27 | 26.4 |
| cobalt-plan | 20 | 0.25 | 0.75 | 0.90 | 32 | 27.6 |
| cobalt-react | 20 | 0.40 | 0.60 | 0.75 | 30 | 28.4 |
| delta-plan | 16 | 0.75 | 0.25 | 1.00 | 24 | 24.4 |
| delta-react | 16 | 1.00 | 0.00 | 1.00 | 24 | 26.4 |
| ember-plan | 16 | 0.00 | 1.00 | 0.50 | - | 25.0 |
| ember-react | 16 | 0.00 | 1.00 | 0.38 | - | 26.2 |
| scripted | 16 | 0.00 | 1.00 | 0.00 | - | 23.2 |

Families span "drifts but self-corrects" (ember) to "almost always violates"
(delta). The scripted control never drifts. Plan-and-execute agents violate
less often than ReAct agents of the same family, but drift about as much,
because committed plans delay the turn from drift to violation.

## Cost profile

About 2,000 agent steps per second on one laptop core for synthetic agents.
At that rate, the blueprint sizing (1.8 million fork steps) takes about 15
minutes before early-stop savings. In the first fork measurements, early
stopping saved about 40% of steps. LLM agents will be orders of magnitude
slower; the cached backend makes reruns free.

## Notes for Phase 4

- Delta and birch reach violation quickly, so their fork grid needs dense,
  early prefixes. Ember rarely violates, so it is mostly a disruption probe.
- Re-run `horizonguard power` with these base rates before full generation.
