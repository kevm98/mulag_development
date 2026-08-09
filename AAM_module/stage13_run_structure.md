# Stage 13 — Run structure

Status: **complete**.

Every final run uses the same matched 11-segment schedule: settle, three
postures, three PRBS identification windows, DLS task motions,
reach-and-return and stow. A production run contains 11,304 control steps or
188.4 simulated seconds.

The length is divisible by the three-control-step ROM period, so every 50 ms
learning transition is exact. Implementation: `aam_data/run_structure.py`.
