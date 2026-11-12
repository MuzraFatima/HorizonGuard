# Baseline leaderboard

| monitor | test MAE | test AUROC | test ECE | test lead | test detect | OOD MAE | OOD AUROC | OOD lead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| probability | 0.133 | 0.968 | 0.019 | 0.500 | 0.500 | 0.183 | 0.907 | 4.500 |
| trajectory-features | 0.274 | 0.770 | 0.064 | 0.000 | 0.400 | 0.251 | 0.805 | 0.000 |
| shortcut-only | 0.335 | 0.651 | 0.053 | 0.000 | 0.000 | 0.312 | 0.689 | 0.000 |
| markov | 0.402 | 0.612 | 0.180 | 0.000 | 0.000 | 0.410 | 0.741 | 0.000 |
| rule | 0.274 | 0.740 | 0.125 | 0.000 | 0.400 | 0.222 | 0.780 | 0.000 |
| e-valuator | 0.133 | 0.968 | 0.019 | 0.000 | 0.400 | 0.183 | 0.907 | 2.000 |
| latent-probe (white-box) | 0.130 | 0.960 | 0.067 | 6.000 | 0.700 | 0.155 | 0.913 | 6.000 |
| risk-controlled |  |  |  | 0.000 | 0.400 |  |  |  |

## Reality check 1: drift beyond shortcuts

- rollout MAE, full features 0.137 vs shortcut-only 0.335; gain 0.198 (95% CI 0.153 to 0.244)
- AUROC full 0.964 vs shortcut 0.651; within-template 0.940 vs 0.692
- verdict: PASS

## Reality check 2: Intervention Paradox (probability alarm, reminder)

| agent | alarm rate | recovery r | disruption d | net benefit |
| --- | --- | --- | --- | --- |
| atlas-plan | 0.00 | 0.000 | 0.000 | 0.0000 |
| atlas-react | 0.05 | 0.000 | 0.000 | 0.0000 |
| birch-plan | 0.58 | 0.113 | 0.071 | 0.0192 |
| birch-react | 0.67 | 0.058 | 0.000 | 0.0317 |
| cobalt-plan | 0.00 | 0.000 | 0.000 | 0.0000 |
| cobalt-react | 0.00 | 0.000 | 0.000 | 0.0000 |
| delta-plan | 0.91 | 0.096 | 0.218 | -0.0606 |
| delta-react | 0.90 | 0.015 | 0.033 | 0.0040 |
| ember-plan | 0.10 | 0.333 | 0.018 | -0.0016 |
| ember-react | 0.03 | 0.000 | 0.000 | 0.0000 |
| scripted | 0.00 | 0.000 | 0.000 | 0.0000 |

- overall net benefit (w = 2.0): -0.0003
- agents hurt by the alarm: delta-plan, ember-plan
- verdict: PASS

## Reading

Generated with `horizonguard baselines --data data/driftbench-cf` on DriftBench-CF v1, default split (ember family and context drift held out as OOD). Lead times are medians in steps at 5 false alarms per 1,000 healthy steps.

- **Drift is visible beyond shortcuts.** Text and action features cut rollout-risk error by about 0.2 against a step-index and clock model, and the gain survives within templates.
- **Probability is well calibrated but late.** The black-box probability monitor tracks rollout risk closely (AUROC 0.97, ECE 0.02), but at the preregistered false-alarm budget it rarely fires early. The white-box latent probe leads by about 6 steps. That gap is the headroom for the Phase 6 hazard model.
- **The Intervention Paradox replicates.** An accurate probability alarm with reminders has overall net benefit of about zero. It helps birch, and it hurts delta-plan, where disruption (0.22) exceeds recovery (0.10). Disruption is non-trivial, so the Phase 5 gate passes, and P2 and P3 stay in scope.
