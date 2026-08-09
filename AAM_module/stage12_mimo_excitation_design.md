# Stage 12 — Independent MIMO excitation

Status: **complete**.

Use deterministic, independent four-channel PRBS only during the three
identification windows. It is injected at the normal action port and stored in
`probe_excitation`, separate from controller output.

Final settings: amplitude `0.02`, hold 10 control steps, posture-specific seeds
`20260812`, `20260813`, and `20260814`. Exact-duration safety flights passed
before the 118-run collection.
