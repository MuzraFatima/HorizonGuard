# Compute and label budget

Forking is the main cost. The number of continuations is

```
continuations = F x G x (1 + ns) x K
agent steps   = continuations x H x (1 - early stop savings)
```

| Symbol | Meaning | Illustrative value |
| --- | --- | --- |
| F | forked runs | 1,000 |
| G | fork-grid prefixes per run | 5 |
| ns | intervention strengths (+1 for the no-intervention arm) | 2 |
| K | continuations per arm | 6 |
| H | continuation horizon in steps | 20 |

The illustrative sizing gives 90,000 continuations and 1.8 million agent steps.
These numbers are placeholders until the Phase 3 pilot measures real step costs
and drift base rates.

Reproduce or explore with:

```bash
horizonguard budget                      # blueprint example
horizonguard budget --k 4 --early-stop 0.3 --stages 4
```

## Policy

- At least 70% of agent steps on small open-weights agents; API agents for spot
  checks and transfer tests.
- Cost controls: short horizon H, common random numbers, early termination of
  forks once the outcome is decided, active allocation of K, cached outputs.
- Spend is capped per phase. Generation is staged, and after every stage the
  stop rule asks whether the label noise already supports the Phase 2 power
  analysis. If it does, generation stops.
- The realised budget is reported in the paper next to this plan.

## Phase caps (placeholder, agree with supervisor)

| Phase | Share of compute |
| --- | --- |
| 3 pilot | 5% |
| 4 DriftBench-CF | 60% |
| 5 to 8 modelling | 15% |
| 9 transfer and stress | 15% |
| 10 closed loop | 5% |
