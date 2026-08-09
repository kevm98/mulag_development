# Stage 19 — No-latent SD baseline

Status: **complete**.

The final protocol-matched baseline is trained from scratch with
`tools/train_sd_z.py --latent-source none`. It uses the dP input mask,
history-6 belief encoder, joint context, unroll 10, normalized state-rollout
loss and hold-start joint rollout.

Final reserved-test result: `R2_all = 0.7989`. This is the required control for
Stage 22, not an older full-observation or mismatched baseline.

Artifact: `../outputs/sd_z_stage22_v6_wide/none_nodp_holdq/`.
