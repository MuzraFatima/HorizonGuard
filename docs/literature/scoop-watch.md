# Scoop-watch log

Weekly queries (fixed in `horizonguard.literature.scoop_watch.QUERIES`):

1. goal drift forecasting
2. agent failure prediction
3. intervention effect agents
4. counterfactual fork rollouts agent
5. conformal or e-process agent monitoring
6. agent monitor transfer

**Triggers.** Any paper that (a) forecasts per-prefix intervention effect for
goal drift, or (b) certifies intervention-induced harm. The keyword screen only
flags candidates; a human read decides.

**Response protocol (within two weeks of a trigger).**

1. Read the paper in full and add a note under `notes/`.
2. Add or update its row in `gap_matrix.yaml` and rerun the validator.
3. Rewrite the affected pillar's difference statement.
4. Decide: keep, sharpen, or pivot along the fallback ladder. Record the decision here.

Run with `horizonguard scoop-watch` (or `--dry-run` to preview). Newest entries first.

<!-- entries -->

## 2026-11-12

Log opened together with the novelty audit in the blueprint (v2). The 19
works in the reference list form the baseline; nothing beyond them has been
logged yet.
