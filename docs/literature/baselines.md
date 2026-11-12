# Baseline reproduction plan

Every baseline runs inside the shared evaluation harness with the same splits
(by agent, scaffold, environment and scenario template, never by step). Where
original code is unavailable we re-implement faithfully, list every deviation
here, and run a sensitivity check on the deviation.

| Baseline | Source | Code | Plan | Module |
| --- | --- | --- | --- | --- |
| Probability-only alarm | Intervention Paradox | n/a | Threshold on the risk model output, same interventions as HorizonGuard | `baselines/probability_alarm.py` |
| SafetyDrift-style Markov monitor | SafetyDrift | request from authors | Absorbing Markov chain over discretised fidelity and tool-level states | `baselines/markov.py` |
| Web-agent macro and micro features | Monitoring Web Agents | request from authors | Logistic model on trajectory summary features plus per-step key-step features | `baselines/trajectory_features.py` |
| E-valuator | E-valuator | public if released | Sequential likelihood-ratio test on the verifier score | `baselines/evaluator.py` |
| Risk-controlled threshold monitor | Online Safety Monitoring | re-implement | Conformal threshold controlling false alarms per 1,000 steps | `baselines/risk_threshold.py` |
| Rule-based drift monitor | common practice | re-implement | Fires when the fidelity score drops below a fixed level for n steps | `baselines/rules.py` |
| Periodic reminders | Drift No More | re-implement | Goal reminder every k steps regardless of risk | `baselines/periodic.py` |
| CIR-style rescue and harm router | CIR | request from authors | Separate rescue and harm scores; intervene when rescue minus harm is positive | `baselines/cir_router.py` |
| Paradox pilot rule | Intervention Paradox | re-implement | Intervene only when the estimated recovery to disruption ratio clears the published rule | `baselines/paradox_rule.py` |
| Activation probes | Goal-Drift Probes | re-implement | Linear probe on hidden states, open-weights agents only, white-box upper bound | `baselines/probes.py` |

## Tolerance for "reproduced"

A baseline counts as reproduced when its headline metric on the original
setting (or our closest analogue) lies within the larger of 2 points or the
reported confidence interval. Otherwise the deviation is documented and the
baseline is still run, flagged as a re-implementation.
