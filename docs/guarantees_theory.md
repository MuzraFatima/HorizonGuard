# Disruption-controlled guarantees: theory note

This note states what HorizonGuard certifies, under which assumptions, and
sketches why the certificates hold. Code lives in `horizonguard.guarantees`.

## Setting

- A **policy** `pi_lambda` acts at a prefix when its score `s(x)` (the best
  predicted net benefit `max_s r_s(x) - w d_s(x)`) is at least `lambda`, and it
  applies the arm `a(x)` attaining that maximum.
- The **loss** at prefix `i` is `l_i(lambda) = 1{s(x_i) >= lambda} * dhat_i(a(x_i))`,
  where `dhat_i(a)` is the fork estimate of the disruption probability for arm `a`:
  the share of seed-paired forks that succeed without the intervention but fail with it.
- **Run-level loss.** For run `j` with prefixes `P_j`,
  `L_j(lambda) = mean_{i in P_j} l_i(lambda)`, which lies in `[0, 1]`.
- **Disruption risk.** `R(lambda) = E[L_j(lambda)]` for a new run drawn from
  the same distribution. This is the expected share of a run's decision points
  at which the policy breaks a run that would otherwise have succeeded.

## Assumptions

1. **Exchangeable runs.** Calibration runs and future runs are exchangeable
   (i.i.d. for the high-probability bound). Prefixes *within* a run need not
   be independent, which is why the unit is the run.
2. **Unbiased labels.** `E[dhat_i(a) | x_i] = d_i(a)`. This holds by the fork
   identification argument (spec, section 7) under complete snapshots and
   seed-determined randomness.
3. **Fixed score function.** The effect model is fitted on training runs that
   are disjoint from the calibration runs.
4. **Monotonicity (for CRC).** `L_j(lambda)` is non-increasing in `lambda`.
   This holds by construction: raising `lambda` only removes actions, and losses
   are non-negative.

## Conformal risk control (expected risk)

Choose `lambda_hat = inf{ lambda : n/(n+1) Rhat_n(lambda) + 1/(n+1) <= alpha }`.

*Sketch.* Under exchangeability of the `n + 1` run losses and monotonicity,
the argument of Angelopoulos et al. (2022) gives
`E[L_{n+1}(lambda_hat)] <= alpha`, where the expectation is over both
calibration and test. Because the label is unbiased (assumption 2), the bound
carries from estimated to true disruption, by the tower property.

*Limit.* It needs `n >= 1/alpha - 1` runs before any action is allowed: 19 runs
at alpha = 0.05 and 99 at alpha = 0.01. Below that the certificate is
**vacuous** (never act), which is safe.

## Learn-then-Test (high probability)

For a grid `lambda_1 > lambda_2 > ...`, ordered from never acting to most
permissive, test `H0_k : R(lambda_k) > alpha` with the Hoeffding-Bentkus
p-value

`p_k = min( exp(-2n (alpha - Rhat_k)_+^2), e * P(Bin(n, alpha) <= ceil(n Rhat_k)) )`

in a fixed sequence, stopping at the first `p_k > delta`. The last rejected
`lambda_k` is selected.

*Sketch.* Each `p_k` is a valid p-value for bounded i.i.d. losses with mean
`R(lambda_k)`. Fixed-sequence testing controls the family-wise error at `delta`
without any correction (Angelopoulos, Bates et al., 2021). So, with
probability at least `1 - delta`, every selected threshold satisfies
`R <= alpha`. Validity does not require monotone risk.

## Time-uniform false alarms

For per-step scores `x_t` in `[0, 1]`, and under the null that a run is
healthy with `E[x_t | F_{t-1}] <= m0`, the wealth
`E_t = prod (1 + kappa_t (x_t - m0))` with predictable `kappa_t` in `[0, 1/m0]`
is a non-negative supermartingale with `E_0 = 1`. By Ville's inequality,
`P(sup_t E_t >= 1/alpha) <= alpha`. Alarming at `1/alpha` therefore bounds the
probability of *any* false alarm in a healthy run, however long it is watched.
`m0` is set to an upper confidence bound on healthy calibration runs.

*Limit.* The conditional-mean assumption is the weak point. Scores rise with
context length even in healthy runs, so `m0` must cover that. The Phase 8
study checks the bound empirically on held-out healthy runs.

## What is not guaranteed

- Nothing transfers automatically to a different agent or environment:
  exchangeability is broken by shift. Phase 9 handles shift with weighted
  calibration and cold-start pooling.
- The certificate bounds *disruption*. It does not bound missed rescues, and
  a vacuous certificate is safe but useless.
- Fork labels are only unbiased for agents whose full state is snapshotted.
