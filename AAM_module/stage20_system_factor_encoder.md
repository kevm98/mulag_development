# Stage 20 — Privileged System Factor Encoder

Status: **complete**.

The encoder learns `z = mu(e)` jointly with `SD_z`. Physical factors are
normalized using training runs only; held-out factors never enter statistics
or control construction. The selected latent dimension is `d_z=2`.

The encoder is privileged and used only to define the target representation.
At deployment, the later AAM must infer `z_hat` from observable history.

Implementation: `tools/train_sd_z.py`.
