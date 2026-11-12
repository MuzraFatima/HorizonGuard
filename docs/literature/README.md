# Literature

| File | Purpose |
| --- | --- |
| `gap_matrix.yaml` | Source of truth for the gap matrix (validated in CI) |
| `gap_matrix.md` | Rendered table, regenerate with `horizonguard gap-matrix --write docs/literature/gap_matrix.md` |
| `notes/` | One structured reading note per work in the reference list |
| `baselines.md` | Reproduction plan for each baseline |
| `scoop-watch.md` | Dated log of the weekly scoop-watch |

Every note starts at *abstract* read level. A row may only be cited once its
note is upgraded to *full-text* and a second reader has checked the matching
row of the gap matrix.
