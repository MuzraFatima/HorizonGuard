# HorizonGuard

[![CI](https://github.com/MuzraFatima/HorizonGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/MuzraFatima/HorizonGuard/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-4f6f41)
![License](https://img.shields.io/badge/license-MIT-b97d12)

**Recoverability-aware early warning of goal drift in long-horizon agents.**

> Not only *will this agent drift?* but *is it still fixable, and will fixing it help more than it hurts?*

Most agent monitors predict failure. But an accurate failure predictor can make agents *worse*: an alarm
that triggers a reminder at the wrong moment can break a run that was going to succeed. HorizonGuard
forecasts, per step, whether a goal re-anchoring intervention would **rescue** the run or **disrupt** it,
acts only when the rescue wins, and **certifies** how often its interventions cause harm.

![HorizonGuard overview](docs/screenshots/overview.png)

## Headline results

On a fresh environment and an agent family that neither model saw in training, in a fork-controlled
closed loop (identical randomness in every arm, 40 seeds):

| Arm | Violation rate | Rescued | Disrupted | Interventions per run |
| --- | --- | --- | --- | --- |
| No monitor | 65% | 0 | 0 | 0.0 |
| Probability-only alarm | 40% | 11 | 1 | 4.0 |
| **HorizonGuard** | **10%** | **25** | 3 | 3.6 |

| Research question | Verdict | Evidence |
| --- | --- | --- |
| RQ1: a multi-state hazard model beats single-event forecasts on lead time | not supported | −1.5 steps vs single-event classifier |
| RQ2: an effect-aware policy beats probability-only alarms | **supported** | net benefit 0.076 vs 0.001 on fresh forks |
| RQ3: the disruption bound is valid and non-vacuous | **supported** | 0 of 200 resamples above alpha; 96% of rescues kept |
| RQ4: drift signatures transfer to unseen agents | **supported** | median AUROC drop 0.03 |
| RQ4: intervention response does not transfer | not supported | effects transferred *better* than risk |
| RQ5: lead time follows a simple scaling law | not supported | inverted U in drift strength |
| RQ5: randomised thresholds blunt stealth drift | **supported** | evasion 100% → 44% |

Negative results are reported as preregistered (`docs/preregistration.md`). All agents are synthetic:
the numbers describe the method, not any real model.

## Demo

| Live monitor | Run replay |
| --- | --- |
| ![Live monitor](docs/screenshots/live-monitor.png) | ![Run detail](docs/screenshots/run-detail.png) |

| Closed-loop A/B | Guarantees |
| --- | --- |
| ![A/B test](docs/screenshots/ab-test.png) | ![Guarantees](docs/screenshots/guarantees.png) |

| Run library | Method | Mobile |
| --- | --- | --- |
| ![Runs](docs/screenshots/runs.png) | ![Method](docs/screenshots/method.png) | ![Mobile](docs/screenshots/mobile.png) |

## How it works

```
Agent ─► trajectory log ─► embeddings ─► multi-state hazard ─► effect model ─► certificate ─► policy
                                             │                     │                 │
                                    aligned / drifting /   rescue vs disruption   disruption risk
                                    violated + recovery    per intervention       ≤ alpha (CRC/LTT)

Fork engine (offline only): snapshot a prefix and branch K seed-paired continuations with and without
each intervention. This gives rollout risk labels plus counterfactual rescue and disruption labels.
```

1. **Counterfactual fork labels (DriftBench-CF).** Every arm shares the same seeds, so each seed shows
   both potential outcomes, and rescue and disruption become measurable per prefix.
2. **Multi-state hazard model.** Onset, violation and recovery are modelled as competing transitions,
   with a risk head supervised by rollout risk.
3. **Effect-aware policy.** For four strengths (reminder, restatement, constraint gate, escalation) the
   policy predicts rescue and disruption, then picks the best `rescue − 2 × disruption`, or waits.
4. **Disruption-controlled guarantees.** Conformal risk control and Learn-then-Test on run-level
   disruption, plus a betting e-process for time-uniform false alarms.
5. **Transfer and stress.** Leave-one-family-out transfer, cold-start recalibration, scaling laws, and
   a gated stealth-drift adversary in simulation.

## Quickstart

```bash
git clone https://github.com/MuzraFatima/HorizonGuard.git
cd HorizonGuard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cd dashboard && npm ci && npm run build && cd ..

horizonguard data          # generate DriftBench-CF (about 4 minutes on a laptop)
horizonguard serve         # API and dashboard on http://127.0.0.1:8765
```

Or regenerate everything with one command: `make all` (data, monitor, every report, paper tables,
dashboard). Docker: `docker build -t horizonguard . && docker run -p 8765:8765 horizonguard`.

## Command line

| Command | What it does |
| --- | --- |
| `horizonguard data` | generate trajectories and fork labels |
| `horizonguard audit` | label-noise audit with replicate forks |
| `horizonguard baselines` | baseline leaderboard and reality checks |
| `horizonguard hazard` | multi-state hazard evaluation |
| `horizonguard effects` | effect learners, policy value on fresh forks, recoverability |
| `horizonguard guarantees` | certificates, coverage, sample complexity, e-process |
| `horizonguard transfer` | transfer, cold start, scaling, stealth stress |
| `horizonguard reproduce` | regenerate every report in `docs/` |
| `horizonguard serve` | run the monitor API and dashboard |
| `horizonguard gap-matrix`, `scoop-watch`, `budget`, `power`, `pilot` | research-management tools |

## Repository layout

```
docs/               specification, preregistration, reports, datasheet, model card
  screenshots/      dashboard screenshots taken by the end-to-end suite
configs/            preregistered constants and generation configs
src/horizonguard/
  envs/ agents/     drift environments and synthetic or LLM agents
  fork/             snapshot, branch, replay, intervention injection
  bench/            DriftBench-CF generation, active labelling, audit, splits
  embeddings/       black-box step encoders and goal-relative geometry
  baselines/        Markov, trajectory features, E-valuator, CIR-style router and more
  hazard/           multi-state hazard model
  effects/          effect learners, policy, recoverability horizon
  guarantees/       conformal risk control, Learn-then-Test, e-process
  transfer/         transfer matrices and cold start
  adversary/        stealth-drift agent (access-controlled)
  monitor/          live monitor, closed loop, HTTP API
dashboard/          React and Vite dashboard, with a Chrome end-to-end suite in e2e/
paper/              drafts of the benchmark and method papers
tests/              unit, statistical-validity and integration tests
```

## Quality

- Python: ruff lint and format, plus 218 tests covering replay determinism, estimators,
  simulation checks of the guarantees and the HTTP API. They run in CI on Python 3.10 and 3.12.
- Dashboard: ESLint, a TypeScript strict build in CI, and a Chrome end-to-end suite (`npm run e2e`)
  that clicks through every page.
- Lighthouse: performance 95, accessibility 100, best practices 100, SEO 100.
- Dashboard performance: route-level code splitting, lazy pages, skeleton loading, debounced search, a
  cached API client, memoised charts, self-hosted fonts, minified bundles. API performance: ETags,
  gzip and pagination.

## Responsible use

All drift is studied in simulation. A monitor reduces risk but does not guarantee safety, and
interventions can themselves cause harm, which is why disruption is measured and bounded on purpose.
The stealth-drift module is gated and simulation-only (`docs/responsible_release.md`).

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
