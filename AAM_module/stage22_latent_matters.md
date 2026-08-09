# Stage 22 — Verify that the latent matters

Status: **STRICT PASS** on 2026-08-09.

The earlier 28-run and 89-run failures were caused by weak factor coverage,
configuration information living mainly in state levels, and full observation
making `z` redundant. The final test uses the 118-run wide matched-excitation
dataset, hides dP, restores causal history/joint context and compares identical
training/rollout protocols.

At H20, paired bootstrap confidence intervals show `z_true` improves both
extension and velocity, pooled and on the 16 reserved test runs, versus:

- zero latent;
- shuffled latent;
- random latent;
- protocol-matched no-latent SD.

All required item-55 checks pass. Mean-latent test velocity is a
diagnostic-only exception and does not change the formal verdict. dz2 passed,
so dz4 was not trained.

Artifacts:

- `../outputs/sd_z_stage22_v6_wide/PASSED_GATE.json`
- `../outputs/sd_z_stage22_v6_wide/dz2_nodp_holdq/gate_verdict.json`
- `../outputs/sd_z_stage22_v6_wide/dz2_nodp_holdq/rollout_gate.json`

Next: Stage 23 may freeze the privileged representation.
