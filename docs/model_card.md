# Model card: HorizonGuard monitor v1.0

## What it is

A black-box monitor for long-horizon agents. For every step it outputs:

- the risk of a goal violation within 20 steps;
- a belief that the agent is drifting;
- the predicted rescue and disruption for four goal re-anchoring strengths;
- a certified decision about whether to intervene, and with which strength;
- a fix-by countdown;
- time-uniform evidence of drift (an e-process).

## Components

| Component | Model | Trained on |
| --- | --- | --- |
| Risk | Multi-state hazard (softmax transition heads) with a rollout-supervised logistic risk head | Step outcomes and fork risk, DriftBench-CF train split |
| Effects | Rescue and disruption decomposition, eight logistic heads | Seed-paired fork labels |
| Abstention | Logistic on "no intervention has positive net benefit" | Fork labels |
| Certificate | Conformal risk control on run-level disruption | Calibration split (29 runs) |
| Evidence | Betting e-process with predictable plug-in bets | Healthy calibration runs (mean bound) |

Inputs are observable only: action names, observation and reasoning text, tool, virtual clock and past interventions.

## Intended use

Research on agent oversight: studying when to intervene on a drifting agent, and how to bound the harm interventions cause. Not intended as a production safety control.

## Evaluation (DriftBench-CF v1, synthetic agents)

| Question | Result |
| --- | --- |
| Effect prediction vs per-agent constant | correlation 0.57 vs 0.43 |
| Policy net benefit on fresh forks | 0.076 vs 0.001 for a probability alarm |
| Closed loop, fresh environment and agent family | violations 60% (no monitor), 33% (probability alarm), 10% (HorizonGuard) |
| Disruption certificate | 0% violations of alpha = 0.05 over 200 resamples; 96% of rescues kept |
| Lead time vs single-event classifier | -1.5 steps (H1 not supported) |
| Stealth drift | 100% evasion against a fixed threshold, 44% with randomised thresholds |

## Limitations

- Every agent is synthetic. The numbers describe the method on simulated families, not any real model.
- Guarantees assume calibration and deployment runs are exchangeable. For a new agent family, pool a few of its runs into calibration first.
- Fork labels are unbiased only when the agent's full state can be snapshotted. API agents with hidden state are excluded from counterfactual claims.
- The text encoder is a hashing bag of words. A sentence-embedding encoder is the obvious upgrade.

## Ethics

Interventions can harm, which is why disruption is measured and bounded on purpose. The stealth-drift adversary is gated and runs only in simulation (see `responsible_release.md`).
