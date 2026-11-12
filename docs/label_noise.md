# DriftBench-CF label-noise report (v1)

Generated with `horizonguard audit --data data/driftbench-cf --replicates 80`
on the default config: 1,179 prefixes, 6 forks per arm.

| arm | mean benefit | noise SE | CI excludes 0 | signal SD | Q | p |
| --- | --- | --- | --- | --- | --- | --- |
| reminder | 0.011 | 0.097 | 0.04 | 0.054 | 1544 | 2.9e-12 |
| restatement | 0.020 | 0.109 | 0.07 | 0.071 | 1673 | 6.3e-20 |
| gate | 0.031 | 0.115 | 0.08 | 0.080 | 1749 | 4.2e-25 |
| escalation | 0.147 | 0.140 | 0.25 | 0.237 | 4564 | 0 |

Replicate agreement on 80 prefixes re-forked with disjoint seeds:

- risk correlation 0.91, noise SD 0.119;
- benefit (escalation) correlation 0.80, noise SD 0.137 against a predicted SE of 0.145.

## Reading

- **Noise is calibrated.** The predicted fork SE matches replicate noise.
  An early version used per-prefix paired SEs, which are zero whenever all
  forks agree. That understated noise by about 40%. The audit now uses the
  unbiased multinomial variance.
- **Heterogeneity is detectable for every strength**, so the Phase 4 gate
  passes. Weak interventions have tiny average effects but real spread. That
  is the setting where an effect model can beat a probability-only alarm.
- **Single-prefix labels are noisy.** Signal SD is at or below noise SE for
  the weaker arms, so Phase 7 must use shrinkage (`bench.active.shrink`) and
  noise-aware losses, with active top-ups for prefixes near the decision
  boundary.
- **The Intervention Paradox is visible already.** About 10% of prefixes have
  negative estimated benefit for the reminder arm.
