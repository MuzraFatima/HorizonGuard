# Gap matrix v1

Generated from `gap_matrix.yaml` with `horizonguard gap-matrix`. Rows marked *abstract* or *summary* in the YAML still need a full-text read and a second reader before citing.

| Work | Target | Signal | Labels | Guarantee | Transfer | Intervention | Leaves open |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arike2025 | drift | black-box | observed | none | none | none | Retrospective measurement only; no forecasting and no intervention. |
| safetydrift | harm | black-box | observed | none | cross-model | none | Hand-designed tool-level states, mostly one model, no intervention effect. |
| goaldrift-probes | drift | white-box | observed | none | none | none | White-box only, one environment, no calibration or intervention. |
| webagent-monitor | failure | black-box | observed | none | cross-site | none | Targets task failure, not goal drift; transfers across sites, not agents. |
| doomed | failure | white-box | observed | miss | none | abort | White-box, in-distribution, optimised to save compute rather than rescue runs. |
| online-safety-monitoring | harm | black-box | observed | false-alarm | none | none | Controls false alarms and misses, not harm caused by intervening. |
| e-valuator | failure | black-box | observed | anytime-valid | none | none | Anytime-valid tests on verifier scores; no intervention-harm control. |
| riskshield | compositional-risk | black-box | observed | false-alarm | cross-model | block | Compositional risk rather than gradual drift; blocks suffixes, no recoverability forecast. |
| streamguard | text-safety | black-box | rollout | none | none | none | Text streams only; no agents and no calibration analysis. |
| intervention-paradox | failure | black-box | observed | none | cross-model | reminder | Diagnoses the paradox with a pilot rule; no per-step effect model, no guarantees. |
| cir | failure | black-box | counterfactual | none | none | refresh | One environment and one model; observation refresh rather than goal drift. |
| drift-no-more | drift | black-box | observed | none | none | reminder | Dialogue user simulator, average effect only, no per-step forecast. |
| distributed-attacks | harm | black-box | observed | none | none | none | Attack side; not a drift-forecasting monitor. |

| Pillar | Closest work | Difference | Novelty | Scoop risk |
| --- | --- | --- | --- | --- |
| P1 Multi-state goal process | safetydrift | Onset, violation and recovery hazards are learned from black-box embeddings and supervised by rollout risk, instead of hand-built absorbing tool states. | 7.0 | medium |
| P2 Recoverability and effect forecasting | cir, intervention-paradox, drift-no-more | Per-prefix effect of graded goal re-anchoring with a fix-by horizon and an abstention option, estimated across agents rather than one model and one environment, and for goal drift rather than observation staleness. | 8.5 | high |
| P3 Disruption control and guarantees | online-safety-monitoring, e-valuator, riskshield | Risk control is applied to counterfactual intervention-induced harm, calibrated with forks, rather than to false alarms or misses. | 8.5 | medium |
| P4 Transfer and cold start | riskshield, webagent-monitor | Leave-one-family and scaffold-out evaluation reported separately for risk, effect and guarantee, with a cold-start recalibration procedure. | 7.5 | medium |
| P5 DriftBench-CF benchmark | streamguard, riskshield, arike2025 | First agent goal-drift benchmark with graded fidelity, onset intervals, rollout-grounded risk and counterfactual intervention effects per prefix. | 8.0 | low |
| P6 Limits and stealth drift | distributed-attacks | Lead-time scaling laws plus an adaptive stealth-drift stress test of the whole monitoring stack, run only inside the simulator. | 6.5 | medium |
