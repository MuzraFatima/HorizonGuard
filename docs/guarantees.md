# Disruption-controlled guarantees

## Held-out disruption (certified on calibration runs, measured on fresh forks)

| method | alpha | lambda | vacuous | held-out risk | rescue | retention | act rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crc | 0.01 | never | True | 0.0000 | 0.0000 | 0.00 | 0.00 |
| crc | 0.02 | never | True | 0.0000 | 0.0000 | 0.00 | 0.00 |
| crc | 0.05 | 0.075 | False | 0.0044 | 0.0772 | 0.96 | 0.38 |
| ltt | 0.01 | never | True | 0.0000 | 0.0000 | 0.00 | 0.00 |
| ltt | 0.02 | never | True | 0.0000 | 0.0000 | 0.00 | 0.00 |
| ltt | 0.05 | never | True | 0.0000 | 0.0000 | 0.00 | 0.00 |
| uncertified | nan | 0.000 | False | 0.0059 | 0.0801 | 1.00 | 0.44 |

## Coverage over 200 resampled calibration splits (alpha = 0.05, delta = 0.1)

- LTT: held-out risk above alpha in 0.0% of resamples (must be at most delta); vacuous in 0%
- CRC: mean held-out risk 0.0102 (must be at most alpha)

## Sample complexity

| calibration runs | LTT vacuous | LTT violations | LTT retention | CRC mean risk |
| --- | --- | --- | --- | --- |
| 20 | 100% | 0.0% | 0.00 | 0.0033 |
| 40 | 100% | 0.0% | 0.00 | 0.0103 |
| 80 | 0% | 0.0% | 0.15 | 0.0099 |
| 120 | 0% | 0.0% | 0.92 | 0.0094 |

## Time-uniform false alarms (betting e-process vs fixed threshold)

- e-process (m0 = 0.539): healthy runs with any false alarm 0.0% (bound 5%), detection 0.58, mean lead 6.35 steps
- fixed threshold at the per-step budget: healthy runs with a false alarm 0.0%, detection 0.50, mean lead 6.12 steps
- held-out healthy runs: 79

## Verdict and reading

Generated with `horizonguard guarantees --data data/driftbench-cf` on DriftBench-CF v1. Assumptions and proof sketches are in [`guarantees_theory.md`](guarantees_theory.md).

- **Valid everywhere.** Across 200 resampled calibration splits, and at every sample size, neither CRC nor LTT ever let held-out disruption exceed alpha = 0.05. Mean CRC risk sits around 0.01.
- **Non-vacuous where data allows.** CRC certifies a real threshold on the default 32-run calibration split and keeps 96% of rescues, with held-out disruption 0.004. The high-probability LTT certificate needs more data: vacuous below about 40 calibration runs, live from about 80, and keeping 92% of rescues at 120. Pooled calibration (half of all non-training runs, 67) is enough for LTT in the coverage study. The gate (a valid, non-vacuous disruption bound that keeps the preregistered 60% of rescues) is **met with CRC**, and **met with LTT once at least 120 calibration runs are pooled**.
- **Tighter alphas (0.01, 0.02) are vacuous on v1** for both methods, because CRC needs at least `1/alpha - 1` runs. This is the honest achievable bound.
- **Time-uniform alarms come close to free.** The betting e-process had no false alarm in any of 79 healthy held-out runs. Its detection (58%) and lead (6.3 steps) slightly beat a fixed per-step threshold, and it adds a guarantee that holds however long a run is watched.
- **Caveat.** These guarantees assume calibration and deployment runs are exchangeable. Shift to new agents is the Phase 9 question.
