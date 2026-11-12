# Transfer, cold start, limits and stealth drift

## Transfer matrix

| axis | held out | risk AUROC drop | risk MAE ratio | effect MAE ratio | guarantee risk (unseen) | holds |
| --- | --- | --- | --- | --- | --- | --- |
| family | atlas | 0.098 | 1.42 | 0.89 | 0.0037 | True |
| family | birch | 0.135 | 3.09 | 2.14 | 0.0150 | True |
| family | cobalt | 0.149 | 0.93 | 0.90 | 0.0007 | True |
| family | delta | 0.100 | 4.60 | 1.74 | 0.0042 | True |
| family | ember | 0.131 | 2.06 | 0.61 | 0.0000 | True |
| family | scripted | nan | 1.18 | 0.47 | 0.0000 | True |
| scaffold | plan | 0.015 | 1.30 | 1.55 | 0.0202 | True |
| scaffold | react | 0.018 | 1.44 | 1.09 | 0.0019 | True |
| scaffold | scripted | nan | 1.18 | 0.47 | 0.0000 | True |
| env | codebase | 0.054 | 1.28 | 1.02 | 0.0208 | True |
| env | portfolio | 0.038 | 1.61 | 1.13 | 0.0158 | True |
| env | research | 0.179 | 2.66 | 1.70 | 0.0191 | True |
| env | sysadmin | 0.011 | 1.44 | 1.05 | 0.0191 | True |
| drift_type | compaction | -0.018 | 1.06 | 1.06 | 0.0071 | True |
| drift_type | context | 0.027 | 1.11 | 0.92 | 0.0082 | True |
| drift_type | distractor | -0.027 | 1.07 | 1.12 | 0.0154 | True |
| drift_type | goal-switch | -0.026 | 1.06 | 1.06 | 0.0117 | True |
| drift_type | nudge | -0.032 | 0.95 | 0.88 | 0.0116 | True |

- H4a (signatures transfer, median AUROC drop 0.033 <= 0.1): SUPPORTED
- H4b (effects degrade more than risk; median MAE ratio effect 1.05 vs risk 1.29): NOT SUPPORTED
- disruption certificate held on 100% of unseen splits

## Cold start (new family, alpha = 0.02)

| family | N target runs | source | target | pooled | weighted | effect MAE |
| --- | --- | --- | --- | --- | --- | --- |
| birch | 0 | vacuous | vacuous | vacuous | vacuous | 0.161 |
| birch | 5 | vacuous | vacuous | rescue 0.012, viol 0% | vacuous | 0.152 |
| birch | 10 | vacuous | vacuous | rescue 0.126, viol 0% | vacuous | 0.153 |
| birch | 20 | vacuous | vacuous | rescue 0.227, viol 3% | vacuous | 0.152 |
| delta | 0 | vacuous | vacuous | vacuous | vacuous | 0.135 |
| delta | 5 | vacuous | vacuous | rescue 0.000, viol 0% | vacuous | 0.141 |
| delta | 10 | vacuous | vacuous | rescue 0.108, viol 0% | vacuous | 0.135 |
| delta | 20 | vacuous | vacuous | rescue 0.194, viol 0% | vacuous | 0.131 |
| cobalt | 0 | vacuous | vacuous | vacuous | vacuous | 0.076 |
| cobalt | 5 | vacuous | vacuous | rescue 0.000, viol 0% | vacuous | 0.082 |
| cobalt | 10 | vacuous | vacuous | rescue 0.001, viol 0% | vacuous | 0.083 |
| cobalt | 20 | vacuous | vacuous | rescue 0.012, viol 0% | vacuous | 0.083 |

## Lead-time scaling

| drift strength | context | violation | detection | mean lead |
| --- | --- | --- | --- | --- |
| 0.5 | 30 | 0.05 | 0.00 | 0.00 |
| 0.5 | 45 | 0.12 | 0.40 | 3.40 |
| 0.5 | 60 | 0.17 | 0.43 | 3.57 |
| 1.0 | 30 | 0.53 | 0.10 | 0.52 |
| 1.0 | 45 | 0.53 | 0.52 | 4.00 |
| 1.0 | 60 | 0.53 | 0.67 | 2.95 |
| 1.5 | 30 | 0.68 | 0.37 | 1.78 |
| 1.5 | 45 | 0.80 | 0.28 | 2.16 |
| 1.5 | 60 | 0.88 | 0.23 | 1.37 |
| 2.0 | 30 | 0.82 | 0.12 | 0.30 |
| 2.0 | 45 | 0.93 | 0.30 | 2.46 |
| 2.0 | 60 | 0.97 | 0.13 | 1.41 |

- preregistered law (log strength, log context): out-of-sample R^2 -0.07 (H5a needs 0.5): NOT SUPPORTED
- exploratory law with (log strength)^2: out-of-sample R^2 -0.24, coefficients -7.54, -0.73, +2.64, -1.18

## Stealth-drift stress test (simulation only)

| setting | violating runs | violation rate | evasion | mean lead |
| --- | --- | --- | --- | --- |
| naive agent | 49 | 0.82 | 0.76 | 2.27 |
| stealth agent | 32 | 0.53 | 1.00 | 0.00 |
| stealth + randomised threshold | 32 | 0.53 | 0.44 | 6.88 |
| stealth + ensemble | 32 | 0.53 | 0.72 | 1.28 |

## Verdict and reading

Generated with `HORIZONGUARD_ENABLE_ADVERSARY=1 horizonguard transfer --data data/driftbench-cf`.

- **H4a supported.** Drift signatures transfer to unseen scaffolds, environments and drift types with a median risk-AUROC drop of 0.03. Unseen *families* are the hard case (drops of 0.10 to 0.15), so a new model family should be treated as a shift.
- **H4b not supported, and this is a genuine surprise.** Intervention response transferred *better* than risk: median effect-error ratio 1.05 against 1.29 for risk. In v1, effects are driven mostly by the drift state and compliance, which the features capture across agents. The blueprint expected agent-specific responses. This is a v1 finding to re-test with LLM agents, whose response to reminders may be far more idiosyncratic.
- **The guarantee survived shift** on all 18 unseen splits at alpha = 0.05. That is partly because base disruption is low. At a tight alpha = 0.02, source-only certificates are vacuous, and **pooling 5 to 10 runs from the new family restores a usable certificate**. The density-ratio-weighted certificate is too conservative to act at v1 sizes, and few-shot effect adaptation adds little.
- **H5a not supported.** Lead time is non-monotone in drift strength. It peaks at moderate drift: weak drift is hard to see, and strong drift violates before a lead can build. Lead grows with context length. Neither the preregistered log-linear law nor an exploratory quadratic predicts held-out cells, so v1 shows no simple scaling law.
- **Stealth drift.** A monitor-aware agent masking near the threshold evades the fixed-threshold monitor every time (lead 0). **Per-run randomised thresholds are the effective defence**: evasion drops to 44%, with a mean lead of 6.9 steps, more than against the naive agent, because the masking itself becomes a signature. An ensemble with a rule monitor helps less (72% evasion). Masking also costs the adversary: its violation rate falls from 0.82 to 0.53. Code is gated; see `responsible_release.md`.
