# Factor-reconstruction sweep — does the SD latent retain what it discards?

Date: 2026-08-16. Result JSON: [`results/factor_recon_sweep.json`](results/factor_recon_sweep.json).
Analysis script: [`factor_recon_sweep.py`](factor_recon_sweep.py).

## Question

Turning on `--factor-recon-weight` in [`AAM_module/tools/train_sd_z.py`](../../AAM_module/tools/train_sd_z.py)
(auxiliary `MSE(decoder(z), e_tilde)` head, default 0.0) should force the 2-D
SD latent to retain all six deployment factors instead of only the ones that
help the dynamics term. We swept `w ∈ {0.0, 0.1, 0.5, 1.0, 2.0}` and measured
(a) what each latent lets a **held-out linear readout** recover per factor and
(b) what that costs the model's own validation loss.

## Protocol

- Five models trained with the exact flags from the handover, **same seed
  `20260807` for every run**, dataset
  `AAM_module/datasets/rom_deployment_stage31_v1_frozen`
  (`--latent-dim 2 --latent-source encoder --unroll 5 --val-unroll 5
  --rollout-q-mode hold-start --rollout-ramp-epochs 40`), one run per weight:
  `outputs/factor_recon_sweep/w0.0|w0.1|w0.5|w1.0|w2.0`.
- `w=0.0` is the baseline control, trained fresh here (the frozen model is not
  a valid control: its `metrics.json` does not record `hidden`, `depth`,
  `dropout`, `lr`, or `seed`).
- Per-run latent: `z = FactorEncoder((e − μ_e)/σ_e)` with the **train-only**
  normalization `μ_e, σ_e` recorded in each run's `metrics.json`.
- Held-out decode: affine fit `z → e` on the **82 train runs**, scored as
  per-factor **R² on the 16 test runs** (factor configs disjoint from train,
  `splits/split_summary.json`: `test_vs_train_leak_keys: 0`; test is 16
  `combined`-domain runs).
- Positive control: seal is known to be strongly encoded; the decode is valid
  only if seal scores held-out R² > 0.5 in the `w=0.0` column.

## Results — held-out R² per factor × factor_recon_weight

R² computed on the test split (16 runs), affine decode trained on the train
split (82 runs).

| w | flow | compliance | seal | pressure | mass_scale | com_z |
|---|---|---|---|---|---|---|
| 0.0 | 0.506 | −0.045 | **0.913** | 0.252 | −0.203 | 0.113 |
| 0.1 | 0.613 | −0.126 | −0.068 | 0.276 | 0.039 | 0.432 |
| 0.5 | 0.637 | −0.111 | −0.089 | 0.311 | 0.067 | 0.389 |
| 1.0 | 0.597 | −0.123 | −0.076 | 0.317 | 0.145 | 0.389 |
| 2.0 | 0.610 | −0.130 | −0.075 | 0.297 | 0.193 | 0.398 |

## Positive control

- seal held-out R² at `w=0.0` = **0.913** (correlation 0.956) → **PASS**
  (threshold > 0.5). The decode procedure is valid.
- In the `w>0` columns seal does **not** pass (≈ −0.08). This is not a decode
  failure — the `w=0.0` control already validated the procedure, and it is
  consistent with these models genuinely no longer exposing seal through a
  linear readout of the 2-D latent (see interpretation below).

## Dynamics cost (model's own validation loss)

`best_val_loss` / `best_epoch` from each run's `metrics.json`, plus the
one-step test metrics for context.

| w | best_epoch | best_val_loss | Δ best_val_loss vs w=0.0 | test r²_all (1-step) | test rmse_all |
|---|---|---|---|---|---|
| 0.0 | 34 | 0.18169 | — | 0.8275 | 4.6745e+05 |
| 0.1 | 38 | 0.17765 | −0.00404 | 0.8175 | 4.8075e+05 |
| 0.5 | 38 | 0.17751 | −0.00418 | 0.8188 | 4.7911e+05 |
| 1.0 | 38 | 0.17795 | −0.00373 | 0.8187 | 4.7921e+05 |
| 2.0 | 38 | 0.17824 | −0.00345 | 0.8180 | 4.8011e+05 |

## Answers to Step 4

1. **Does any weight make `mass_scale` decode at held-out R² > 0.5? No.** The
   best linear held-out mass_scale R² across the sweep is **0.193 at w=2.0**
   (values: −0.203, 0.039, 0.067, 0.145, 0.193); no weight reaches 0.5.
2. **What does that cost in `best_val_loss` relative to `w=0.0`? Nothing — it
   slightly improves.** `best_val_loss` drops from 0.18169 (w=0.0) to
   0.17751–0.17824 (w=0.1–2.0), i.e. Δ in [−0.00418, −0.00345]. The one-step
   test R² does drop a little (0.8275 → 0.8180), so the trade-off is a small
   one-step test-R² dip, not a `best_val_loss` penalty.
3. **Did the seal positive control pass in every column?** The mandatory
   control — seal > 0.5 in the `w=0.0` column — **passed (0.913)**, validating
   the decode. It does **not** hold in the `w>0` columns (≈ −0.08): those
   models genuinely stopped exposing seal through a linear latent readout, a
   substantive result rather than a decode fault.

## Interpretation

- **The recon loss does spread the 2-D budget.** Versus the w=0.0 latent
  (seal 0.913, everything else ≤ 0.51), the w>0 latents make flow (≈0.60),
  com_z (≈0.39) and pressure (≈0.28–0.32) linearly recoverable and lift
  mass_scale from −0.20 to at most 0.19.
- **It does not make mass linearly legible, and it destroys the seal channel.**
  With only 2 latent dimensions the reconstruction term forces a compromise:
  the previously-dominant seal encoding collapses to ≈ −0.08 while mass — the
  factor the handover hypothesised was dropped because it cost dynamics — still
  never reaches a useful linear R². The hypothesis that a reconstruction term
  would "retain the factors it currently discards" is only partially realised,
  and not for mass.
- **No `best_val_loss` cost was observed.** The handover's premise that keeping
  the discarded factors costs dynamics accuracy (in `best_val_loss`) is not
  reproduced here; the recon runs have slightly *lower* `best_val_loss` but a
  small one-step test-R² drop and a materially worse latent for the one factor
  (seal) the w=0.0 latent encoded well.

## Secondary measurement (transparency)

`metrics.json` also stores the trained nonlinear decoder's own reconstruction
R² on held-out val+test runs (`factor_recon` field, evaluated on normalized
factors). By that (non-linear) measure mass_scale reaches R² 0.531 at w=0.1 and
0.483 at w=2.0, and seal stays ≈ 0.28–0.32. The required **held-out linear**
readout — the contract PPO/ID actually consume, per
[`experiments/README.md`](README.md) ("the factors must be linearly legible in
z itself") — does not recover mass above 0.193 at any weight. The two
measurements agree that seal is no longer the dominant factor once recon is on;
they differ only on how much a *nonlinear* head can still pull out of the same
2-D latent.

## Reproducibility

- Training command for each weight (w=0.0 run separately, w=0.1/0.5/1.0/2.0 in
  one loop), all exit 0, each ~2 min CPU, Python 3.11.15
  (`/home/rrlab/miniconda3/envs/env_isaaclab/bin/python`).
- Every number above is read from either
  `outputs/factor_recon_sweep/w*/metrics.json` or produced by
  [`factor_recon_sweep.py`](factor_recon_sweep.py) (which writes
  [`results/factor_recon_sweep.json`](results/factor_recon_sweep.json),
  including the per-run decoded and true factor vectors).
