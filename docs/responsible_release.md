# Responsible-release decision log

| Date | Artefact | Decision | Reason |
| --- | --- | --- | --- |
| 2026-11-12 | DriftBench-CF v1 environments and labels | Release | Simulated office tasks; violations are symbolic actions with no operational uplift |
| 2026-11-12 | `horizonguard.adversary` (stealth-drift agent) | Gated: import requires `HORIZONGUARD_ENABLE_ADVERSARY=1`; publish metrics only | The agent only exploits HorizonGuard's own simulated monitors, but it demonstrates a monitor-aware masking strategy. Results report evasion rates and defences, not tuned evasion recipes |

## Rules for the stealth-drift study

- Runs only inside the simulator, against HorizonGuard monitors.
- No prompts or fine-tuning recipes for real LLM agents are written or released.
- Papers report evasion rate, lead-time erosion and defence effectiveness.
- Any extension to LLM agents needs a new entry in this log before code is written.
