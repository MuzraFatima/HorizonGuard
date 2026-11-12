# Paper A outline: DriftBench-CF

**Working title.** DriftBench-CF: Counterfactual Fork Labels for Goal Drift in
Long-Horizon Agents.

**Venue.** Datasets and benchmarks track. Preprint around week 20, submission
after Phase 5.

## 1. Introduction

- Monitors predict failure, but deployment needs to know whether intervening
  helps.
- Observed outcomes cannot answer that question; counterfactual forks can.
- Contributions: fork-label methodology, benchmark, noise audit, baselines.

## 2. Related work

Gap matrix rows: SafetyDrift, web-agent monitors, StreamGuard, CER-Bench, Arike
et al., CIR, the Intervention Paradox.

## 3. Estimands and identification

Potential outcomes under seed-paired forks; rescue and disruption; the
recoverability horizon; assumptions and where they fail (spec, sections 4 to 7).

## 4. Benchmark construction

Environments, agents, drift types, fork grid, staged generation, budget.

## 5. Label quality

Replicate agreement, calibrated noise, heterogeneity tests, how many forks are
needed (power analysis plus a K ablation).

## 6. Baselines

Phase 5 leaderboard: probability-only alarm, Markov monitor, trajectory
features, E-valuator, risk-controlled threshold, periodic reminders,
CIR-style router.

## 7. When intervening hurts

Disruption by family and strength; replication of the paradox.

## 8. Limitations

Synthetic v1 agents; snapshot assumptions for API models; simulated tasks.

## Figures

1. Fork diagram (same prefix, branched arms).
2. Risk vs benefit scatter coloured by family.
3. Benefit vs prefix step with the recoverability horizon.
4. Replicate agreement.
5. Disruption by family and intervention strength.
