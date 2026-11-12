# Multi-state hazard model

## Lead-time curve (test + OOD, mean lead in steps, misses count as 0)

| monitor | lead@1 | lead@2 | lead@5 | lead@10 | lead@20 | detect@5 | MAE | ECE | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi-state-hazard | 2.38 | 5.77 | 7.88 | 10.15 | 11.58 | 0.58 | 0.199 | 0.129 | 0.908 |
| probability | 3.35 | 5.00 | 8.81 | 10.04 | 13.73 | 0.77 | 0.169 | 0.081 | 0.928 |
| single-event | 3.69 | 5.04 | 9.35 | 11.50 | 12.50 | 0.65 | 0.190 | 0.119 | 0.906 |
| logistic-hazard | 3.69 | 4.38 | 5.31 | 7.04 | 8.77 | 0.50 | 0.190 | 0.073 | 0.854 |
| autoencoder-anomaly | 2.12 | 4.15 | 5.81 | 5.81 | 11.54 | 0.58 | 0.504 | 0.467 | 0.703 |

## Gain over each comparator at 5 false alarms per 1,000 steps

- vs probability: -0.92 steps (95% CI -2.23 to +0.35)
- vs single-event: -1.46 steps (95% CI -3.00 to -0.23)
- vs logistic-hazard: +2.58 steps (95% CI -0.27 to +5.27)
- vs autoencoder-anomaly: +2.08 steps (95% CI -0.08 to +4.58)

H1 (lead-time gain over the best baseline): NOT SUPPORTED

## Ablations

| monitor | lead@1 | lead@2 | lead@5 | lead@10 | lead@20 | detect@5 | MAE | ECE | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi-state-hazard | 2.38 | 5.77 | 7.88 | 10.15 | 11.58 | 0.58 | 0.199 | 0.129 | 0.908 |
| hazard: actions only | 0.50 | 1.19 | 2.73 | 4.88 | 5.27 | 0.23 | 0.268 | 0.124 | 0.784 |
| hazard: actions + text | 4.04 | 4.58 | 8.62 | 11.23 | 13.27 | 0.65 | 0.200 | 0.128 | 0.910 |
| hazard: + geometry (no shortcuts) | 3.50 | 4.65 | 8.46 | 11.88 | 13.73 | 0.65 | 0.198 | 0.125 | 0.913 |
| hazard: realised labels only | 3.50 | 5.81 | 8.65 | 11.15 | 12.00 | 0.65 | 0.191 | 0.121 | 0.906 |

## Early signature (standardised coefficients)

- **onset**: mean |coef| by group: act 0.25, emb 0.13, geo 0.31, short 0.58; top: geo:angle_ema (+0.90), act:bucket_13 (-0.72), short:t (-0.70), act:tool_shell (+0.67), act:tool_browse (-0.60)
- **violation**: mean |coef| by group: act 0.15, emb 0.14, geo 0.26, short 0.39; top: short:log_t (+0.84), act:bucket_9 (-0.65), emb:54 (-0.59), emb:60 (-0.59), geo:dist_ema (+0.47)

## Verdict and reading

Generated with `horizonguard hazard --data data/driftbench-cf` on DriftBench-CF v1: 26 violating runs in the pooled test and OOD splits.

- **H1 is not supported on v1.** At the preregistered budget, the multi-state model trails the single-event classifier by 1.5 steps of mean lead (CI excludes zero), and it ties the probability monitor. It beats the logistic-hazard and autoencoder comparators by 2 to 2.6 steps, although those intervals touch zero. Under the preregistration, H1's kill criterion is met for v1 data. The claim is retired for synthetic agents and re-tested when LLM agents are added.
- **Why, most likely.** Composing per-step hazards under fixed current features over 20 steps is a strong assumption, and drifting-to-violated transitions are sparse (about 70 in training). The state structure adds parameters without adding information the single-event head lacks. The interpretable outputs (onset hazard, drifting belief, fix-by horizon inputs) are still used by Phases 7 and 8.
- **Text carries the early signal.** Actions alone give a mean lead of 2.7 steps. Adding the reasoning text gives 8.6. Shortcut features add nothing: dropping step index and clock does not hurt.
- **Early signature.** Onset is driven most by the goal-relative angle of the smoothed embedding (`geo:angle_ema`) and by tool mix. Violation is driven by elapsed steps and by distance from the anchor.
- **Rollout supervision** improves calibration against fork risk, but not lead time.

Deviation logged in the preregistration: the causal encoder is the hand-built featurizer rather than a learned Transformer, TCN or state-space model.
