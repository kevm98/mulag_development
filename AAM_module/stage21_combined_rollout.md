# Stage 21 — Privileged latent plus SD training

Status: **complete**.

Final dz2 training uses:

- teacher-forced one-step delta loss;
- normalized autoregressive state-rollout loss;
- rollout weight ramp over 60 epochs;
- dP-hidden deployment mirror;
- causal history-6 belief and joint context;
- unroll horizon 10 and hold-start joint rollout.

The matched dz2 model reached reserved-test `R2_all = 0.8186` at best epoch
252, versus `0.7989` for the no-latent control.

Artifact: `../outputs/sd_z_stage22_v6_wide/dz2_nodp_holdq/`.
