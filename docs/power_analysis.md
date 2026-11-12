# Power analysis

Simulation-based, from `horizonguard.eval.power`. Regenerate with
`horizonguard power --sims 200 --write docs/power_analysis.md`.

## Question

How many seed-paired forks per arm (K) and how many prefixes do we need to
detect (a) a non-zero mean intervention benefit and (b) heterogeneity in
benefit across prefixes? (b) is what the effect model in Phase 7 needs. If
effects do not vary beyond fork noise, there is nothing to forecast.

## Default scenario

- Baseline violation risk around 0.5, varying across prefixes (Beta).
- 70% helpful prefixes (rescue 0.25, disruption 0.02) and 30% harmful
  prefixes (rescue 0.02, disruption 0.15).
- Tests at alpha = 0.05. 200 simulated datasets per cell.

| K | prefixes | mean-effect power | heterogeneity power | per-prefix MAE |
| --- | --- | --- | --- | --- |
| 2 | 50 | 0.69 | 0.41 | 0.269 |
| 2 | 100 | 0.94 | 0.62 | 0.268 |
| 2 | 200 | 0.99 | 0.78 | 0.270 |
| 2 | 400 | 1.00 | 0.89 | 0.268 |
| 4 | 50 | 0.81 | 0.73 | 0.176 |
| 4 | 100 | 0.99 | 0.90 | 0.174 |
| 4 | 200 | 1.00 | 0.99 | 0.174 |
| 4 | 400 | 1.00 | 1.00 | 0.174 |
| 6 | 50 | 0.91 | 0.92 | 0.146 |
| 6 | 100 | 0.99 | 1.00 | 0.147 |
| 6 | 200 | 1.00 | 1.00 | 0.147 |
| 6 | 400 | 1.00 | 1.00 | 0.147 |
| 8 | 50 | 0.94 | 0.99 | 0.122 |
| 8 | 100 | 1.00 | 1.00 | 0.121 |
| 8 | 200 | 1.00 | 1.00 | 0.121 |
| 8 | 400 | 1.00 | 1.00 | 0.122 |
| 12 | 50 | 0.97 | 1.00 | 0.102 |
| 12 | 100 | 1.00 | 1.00 | 0.102 |
| 12 | 200 | 1.00 | 1.00 | 0.102 |
| 12 | 400 | 1.00 | 1.00 | 0.102 |

## Calibration check (no heterogeneity)

With every prefix sharing the same effect, the heterogeneity test should reject
about 5% of the time:

| K | prefixes | false-positive rate |
| --- | --- | --- |
| 2 | 100 | 0.16 |
| 4 | 100 | 0.10 |
| 8 | 100 | 0.08 |

At K = 2 the chi-square approximation breaks down on sparse counts, so the
test is anti-conservative there.

## Decisions

- **K = 6 is the minimum for counterfactual labels**, and K = 4 the floor for
  any heterogeneity claim. K = 6 matches the blueprint sizing and reaches 92%
  heterogeneity power with only 50 prefixes.
- Per-prefix benefit estimates stay noisy (MAE around 0.15 at K = 6), so
  Phase 7 must use shrinkage and noise-aware losses, as the blueprint plans.
- These numbers come from an assumed scenario. They are re-run on the Phase 3
  pilot base rates before DriftBench-CF generation starts.
