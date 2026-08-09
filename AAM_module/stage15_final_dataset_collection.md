# Stage 15 — Final deployment collection

Status: **passed**.

The final manifest contains 118 matched-excitation runs:

| Domain | Runs |
|---|---:|
| nominal train | 2 |
| hydraulic train | 40 |
| mechanical train | 12 |
| combined train | 48 |
| combined test | 16 |

`tools/collect_run_structure.py` collected four configurations per Isaac
process with per-environment factors and logging. Slot-specific safety failures
were discarded and recollected in environment 0. The final collection gate
passed all 118 runs with no issues.

Artifacts: `datasets/master_v3_stage22_wide/` and
`results/stage22_wide_collection_gate.json`.
