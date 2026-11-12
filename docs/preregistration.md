# Preregistration

Frozen on 2026-11-12 before any DriftBench-CF data was generated. Numeric
constants live in [`configs/preregistration.yaml`](../configs/preregistration.yaml)
and are loaded by the code, so the analysis cannot silently drift from this
document.

## Principle

Decide the claim before the data does. Each pillar has a kill criterion; when a
kill criterion is met, the pillar is retired and the fallback ladder applies.

## Research questions and hypotheses

### RQ1 Structure (Pillar 1)

- **H1.** A learned multi-state hazard model (aligned, drifting, violated, with
  recovery) gives a longer median lead time at a false-alarm budget of 5 per
  1,000 steps than the best of: Markov-state monitor, single-event classifier,
  discrete-time logistic hazard, autoencoder anomaly detector.
- **H1b.** Its error against rollout-grounded risk is at most 0.05.
- **Primary metric.** Median lead time at budget; mean absolute error against
  the Monte Carlo violation probability from forks.
- **Kill.** Lead-time gain over the best baseline has a 95% bootstrap interval
  that includes zero on seen agents.

### RQ2 Recoverability (Pillar 2)

- **H2.** Per-prefix intervention effect and the recoverability horizon are
  predictable from black-box signals.
- **H2b.** An effect-aware policy has positive net benefit (interval excludes
  zero) in at least 50% of agent-environment cells where the probability-only
  alarm has net benefit at or below zero.
- **Primary metric.** Net benefit = rescue rate minus 2.0 x disruption rate;
  policy value estimated with fresh forks.
- **Kill.** H2b fails, or effect estimates are not better calibrated than a
  constant per-agent effect. Fallback rung L1.

### RQ3 Guarantees (Pillar 3)

- **H3.** Calibrating the alert threshold on fork-labelled runs bounds
  disruption risk at alpha = 0.05 with confidence 0.9, with empirical coverage
  near nominal on held-out runs, while retaining at least 60% of rescues.
- **Primary metric.** Held-out disruption risk vs target; rescues retained.
- **Kill.** The bound is vacuous (no threshold certifiable) or held-out
  disruption exceeds alpha in more than delta of resampled calibration sets.
  Fallback rung L2.

### RQ4 Transfer (Pillar 4)

- **H4a.** Drift signatures transfer to unseen agent families with an AUROC
  drop of at most 0.10.
- **H4b.** Intervention response does not transfer: effect calibration degrades
  significantly more than risk calibration.
- **H4c.** A cold-start sample of N healthy runs and M fork pairs restores the
  disruption guarantee; we report the smallest N and M that do.
- **Kill.** Neither H4a nor H4c holds; reported as a rigorous negative.

### RQ5 Limits (Pillar 6)

- **H5a.** Lead time follows a fitted law in drift strength and context length
  with out-of-sample R-squared of at least 0.5.
- **H5b.** A stealth-drift adversary reduces lead time by a measurable amount;
  we report evasion rate with and without randomised thresholds and ensembles.

## Analysis plan

- Metrics computed per run first, then aggregated; bootstrap intervals over
  runs (2,000 resamples), with Monte Carlo fork error propagated into every
  effect estimate.
- Splits by agent, scaffold, environment and scenario template, never by step.
- At least 5 seeds for every learned model.
- Holm correction across the comparisons inside each RQ; effect sizes reported
  alongside p-values.
- Shortcut controls (step index, trajectory length, topic-matched healthy runs)
  in every evaluation.

## Fallback ladder

| Rung | Condition | What remains |
| --- | --- | --- |
| L0 | P2 and P3 both work | Full plan, P1 to P6 |
| L1 | P2 scooped or effects null | When intervening hurts: disruption analysis, P3, P4, P5 |
| L2 | Guarantees vacuous | Effect-aware policy, benchmark, transfer study |
| L3 | Drift too rare | Stress-test methodology, benchmark, scaling laws |

## Deviations

Every change to this document or to `configs/preregistration.yaml` after the
freeze date is listed here with date, reason and expected impact.

| Date | Change | Reason | Impact |
| --- | --- | --- | --- |
| none yet | | | |
