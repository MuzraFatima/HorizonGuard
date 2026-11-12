# Recoverability and intervention effects

## Effect learners (test + OOD)

| learner | corr | slope | binned error | MAE |
| --- | --- | --- | --- | --- |
| t-learner | 0.537 | 0.613 | 0.042 | 0.088 |
| decomposition | 0.565 | 0.710 | 0.027 | 0.078 |
| direct-shrunk | 0.523 | 0.658 | 0.031 | 0.081 |
| dr-observational | 0.188 | 0.118 | 0.163 | 0.196 |
| constant per agent | 0.430 | 0.657 | 0.030 | 0.091 |

## Policy value on fresh forks (net benefit, w = 2.0, 559 prefixes)

| policy | value | 95% CI | intervene rate |
| --- | --- | --- | --- |
| effect-aware | 0.0760 | 0.0494 to 0.1059 | 0.44 |
| probability alarm (reminder) | 0.0009 | -0.0054 to 0.0071 | 0.23 |
| probability alarm (escalation) | 0.0602 | 0.0347 to 0.0878 | 0.23 |
| CIR-style router | -0.0030 | -0.0115 to 0.0052 | 0.37 |
| Paradox rule | 0.0045 | -0.0030 to 0.0120 | 0.40 |
| periodic reminders | -0.0075 | -0.0127 to -0.0024 | 0.42 |
| never intervene | 0.0000 | 0.0000 to 0.0000 | 0.00 |

## RQ2 gate (agent x environment cells)

- cells where the probability alarm is at or below zero: 35
- effect-aware value positive with CI excluding zero: 12 (34%, target 50%)
- effect-aware no worse than doing nothing: 32
- verdict: NOT MET

## Recoverability horizon (fix-by)

- point: recoverable agreement 0.67, mean absolute error 4.59 steps, signed +4.59, early or exact 0.50
- conservative: recoverable agreement 0.67, mean absolute error 4.39 steps, signed +3.98, early or exact 0.52

## When intervening hurts

- share of effect variance explained by compliance: reminder 0.27, restatement 0.36, gate 0.37, escalation 0.64
- mean disruption by agent (reminder, restatement, gate, escalation):
  - atlas-plan: 0.018, 0.018, 0.010, 0.013
  - atlas-react: 0.004, 0.004, 0.000, 0.000
  - birch-plan: 0.073, 0.068, 0.066, 0.070
  - birch-react: 0.007, 0.017, 0.034, 0.020
  - cobalt-plan: 0.046, 0.048, 0.059, 0.075
  - cobalt-react: 0.026, 0.042, 0.036, 0.010
  - delta-plan: 0.050, 0.056, 0.050, 0.044
  - delta-react: 0.003, 0.003, 0.006, 0.012
  - scripted: 0.000, 0.000, 0.000, 0.000

## Verdict and reading

Generated with `horizonguard effects --data data/driftbench-cf` on DriftBench-CF v1. Policy values use **fresh forks** (disjoint seeds), so no policy is scored on the labels it was fitted or selected on.

- **H2 (effects are predictable): supported.** The rescue and disruption decomposition learner beats the preregistered kill comparison, a constant per-agent effect: correlation 0.57 vs 0.43, MAE 0.078 vs 0.091. Its effect slope is 0.71, so effects are mildly over-dispersed and need shrinkage, as planned.
- **Forks matter.** A DR-learner fitted on one randomly assigned arm per prefix, the observational-log setting, collapses to a correlation of 0.19.
- **The effect-aware policy wins overall.** Net benefit is 0.076 (CI 0.049 to 0.106) against 0.001 for the probability alarm with reminders, the setting the Intervention Paradox warns about. CIR-style routing, the Paradox rule and periodic reminders are at or below zero. A fixed escalation alarm is the strongest competitor (0.060), and the effect-aware policy gains on it mainly by choosing *which* intervention to use.
- **H2b per-cell gate: not met.** The effect-aware policy is strictly better in 34% of the cells where the probability alarm fails (target 50%), and no worse in 32 of 35. Cells hold 4 to 10 prefixes, so per-cell intervals are wide. The gate is reported as preregistered, and the fallback ladder stays at L0 because the pooled result is strongly positive.
- **Fix-by horizon.** It is right about recoverability for 67% of runs and errs by about one fork-grid interval. When wrong, it errs *late*, so the dashboard shows the countdown with that caveat. A denser fork grid near the horizon is the fix.
- **When intervening hurts.** Disruption concentrates in plan-and-execute agents (birch-plan, cobalt-plan, delta-plan), where re-anchoring forces a replan. Compliance explains 27% (reminder) to 64% (escalation) of the variance in effect.
