# HorizonGuard

[![CI](https://github.com/MuzraFatima/HorizonGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/MuzraFatima/HorizonGuard/actions/workflows/ci.yml)

**Recoverability-aware early warning of goal drift in long-horizon agents.**

> Not only *will this agent drift?* but *is it still fixable, and will fixing it help more than it hurts?*

HorizonGuard is a research codebase for monitoring long-horizon agents. It
forecasts goal drift from black-box trajectories, estimates per prefix whether a
goal re-anchoring intervention would still rescue the run or would break a run
that was going to succeed, and only raises alerts whose disruption risk is
certified on fork-labelled calibration data.

## Thesis

Existing work already forecasts agent failure, probes activations, and controls
false-alarm rates. What is missing is the space between those works:

| Gap | What HorizonGuard adds |
| --- | --- |
| Recoverability | Per-prefix forecasts of intervention effect and a *fix-by* countdown |
| Guarantees on the right risk | Certified bounds on intervention-induced disruption, not only false alarms |
| Transfer | Separate transfer studies for risk, effect and guarantee on unseen agents |
| Ground truth | Counterfactual fork labels instead of observed outcomes only |
| Structure | Drift onset, violation and recovery modelled as states with learned transitions |

## Research questions

| RQ | Hypothesis | Primary metric |
| --- | --- | --- |
| RQ1 Structure | A learned multi-state hazard model forecasts onset and violation earlier and better calibrated than Markov states or single-event classifiers. | Lead time at fixed false-alarm budget; error vs rollout risk |
| RQ2 Recoverability | Per-prefix intervention effect and the recoverability horizon are predictable from black-box signals; effect-aware policies beat probability-only alarms. | Net benefit; policy value |
| RQ3 Guarantees | Fork-labelled calibration certifies a disruption-risk bound with valid finite-sample coverage while keeping most rescues. | Empirical coverage; rescues retained |
| RQ4 Transfer | Drift signatures transfer to unseen agents; intervention response does not; a small cold-start sample restores the guarantee. | Transfer gap per component; samples needed |
| RQ5 Limits | Lead time follows a predictable law in drift strength and context length; a stealth-drift adversary erodes it measurably. | Scaling fit; evasion rate |

## Architecture

```
Agent ──► Trajectory logger ──► Embedding layer ──► Multi-state hazard ──► Effect model
                                                                              │
          Dashboard + human ◄── Intervention policy ◄── Guarantee layer ◄─────┘

Fork engine (offline): snapshot a prefix, branch K continuations with and
without intervention → rollout risk labels and counterfactual effect labels.
```

The monitor only reads logged, observable behaviour. The fork engine is an
offline research instrument and is never needed at deployment time.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make lint test
```

## Repository layout

```
docs/               specification, literature, preregistration, data and model cards
configs/            experiment configs (YAML)
src/horizonguard/   library code, one subpackage per stage
tests/              replay determinism, metrics, guarantees
dashboard/          web UI for the live monitor
paper/              LaTeX sources
```

## Roadmap

| Phase | Focus |
| --- | --- |
| 0 | Repository foundation, CI, templates |
| 1 | Novelty-locked framing, scoop-watch, budget |
| 2 | Formal framework, estimands, labelling scheme |
| 3 | Environments, agents, fork engine |
| 4 | DriftBench-CF counterfactual labels |
| 5 | Representations, baselines, reality checks |
| 6 | Multi-state hazard model |
| 7 | Recoverability and intervention-effect forecasting |
| 8 | Disruption-controlled guarantees |
| 9 | Transfer, cold start, limits, stealth-drift stress |
| 10 | Closed loop, reproduction, release |

## Responsible use

All drift is studied in controlled simulations. A monitor reduces risk but does
not guarantee safety, and interventions can themselves cause harm, which is why
disruption is measured on purpose. The stealth-drift module stays
simulation-only and access-controlled.

## License

MIT, see [LICENSE](LICENSE).
