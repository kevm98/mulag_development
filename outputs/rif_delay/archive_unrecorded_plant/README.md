# Reports whose plant configuration is unrecorded

These are the ORIGINAL method-R1 scorings of seeds 17 and 101 and the
history-zeroed control, produced 2026-08-30.

They are archived, not deleted: they are the evidence behind the previously
reported result (+120.1% cut / +78.2% lateral for seed 17).

## Why they were set aside

`PPO_ROM/scripts/evaluate_gates.py` was modified 2026-08-31 16:34 UTC to record
the plant configuration into the report. EVERY report written before that edit
is missing the keys entirely -- `deployment_mass_bimodal`,
`deployment_mass_step_scale`, `deployment_mass_step_time_s`,
`payload_scale_range`, `deployment_mass_step_seed` are absent, not null.

So for these runs there is **no artefact that records which plant they were
scored on**. The producing scripts do export the flip variables, but at least
one of them (`score_arms_gate.sh`, which writes `g_base`) has an mtime LATER
than the report it produced, so the script on disk is not evidence of what ran.

They were therefore not comparable, with certainty, against arms scored after
the edit. Rather than assume, every arm was rescored on the declared plant with
the current code, and the reported table is built entirely from those.

Do not mix these files into a table with post-edit reports.
