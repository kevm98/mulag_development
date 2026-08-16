# Stage 26 — Train and evaluate AAM

Status: **complete; temporal-consistency retraining passed the downstream
causal gate**.

The AAM was trained from scratch on training runs and early-stopped on
validation. The objective is supervised latent MSE plus a causal
adjacent-window temporal-consistency MSE, with weight 1.0. No future window is
used by either the objective or deployment inference.
Evaluated latent RMSE, temporal stability and held-out behavior, with
downstream `SD_z(s, u, z_hat)` rollout quality as the primary metric. The
downstream score uses the prediction available at the causal window ending at
each rollout start; the old full-run mean-latent score is superseded.

Corrected causal results (held-out):

- selected L5 latent test RMSE 1.251 (per dim 1.00 / 1.46); all four tested
  histories were acceptable and the shortest-history rule retained L5;
- downstream H20 rollout: true ext 0.002481 / vel 0.025481; **causal zhat
  0.002524 / 0.025493**; zero 0.003719 / 0.028596; shuffled 0.002729 /
  0.026321; mean 0.002648 / 0.026089; no-latent 0.002656 / 0.025931.
- with the temporal-trained L5 and past-only rolling-20 smoother, the H20
  causal chain reaches the Stage 27 gate: **0.002512 / 0.025436** on the
  pooled held-out rollout report.

The temporal-trained causal z_hat beats the no-latent control with pooled CIs
fully below zero (extension -0.000193..-0.000098, velocity
-0.000717..-0.000281), and the z_hat-vs-z_true CIs contain zero on both
channels. Retention is 0.825 extension / 1.101 velocity. Corrected artifacts
are under `../outputs/aam_stage25_temporal/` and
`../outputs/aam_stage27_temporal/`.

Artifacts: `../outputs/aam_stage26_causal/` (causal rollout reports and
`zhat_vs_none_h20.json`). The prior `../outputs/aam_stage26/` report is
historical noncausal evidence. Tools: `tools/run_stage26_aam.sh`,
`tools/train_aam.py`, `tools/predict_aam.py`, `tools/report_aam_stage26.py`.
