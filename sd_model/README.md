# SD Model — Stage 19 No-Latent Baseline

Standalone copy of the Stage 19 SD baseline (`SD_0(s_t, u_t) -> Delta s_{t+1}`),
trained from scratch on GPU from a real multi-domain Isaac dataset. This folder
lives outside `AAM_module` on purpose: it is the model artifact directory
(checkpoints, metrics, logs, plots), while `AAM_module` keeps the pipeline
(collectors, dataset builders, trainer source, stage docs).

## Contents

```text
checkpoints/<mode>.pt   best checkpoint per input mode (MLP 3x256, SiLU, dropout 0.05)
metrics/<mode>.json     final RMSE, R2, z-score stats, per-epoch history
logs/<mode>.log         per-epoch training log (loss + val RMSE)
data/zero_baseline.json zero-delta predictor RMSE on the held-out test run
data/parity_hist3.npz   one-step predictions vs targets on test (best mode)
plots/                  training plots (see below)
make_plots.py           reproducible plot generator
```

## Models

All models consume observable inputs only (never the privileged factor vector
`z_i`):

| mode | input | dim | test R2 vs zero-delta |
| --- | --- | ---: | ---: |
| `su` | `[s_t^H; u_t]` | 20 | 0.815 |
| `su_delta` | `[s_t^H; u_t; s_t^H - s_{t-1}^H]` | 32 | 0.888 |
| `squ` | `[s_t^H; q_t, qdot_t; u_t]` | 30 | 0.820 |
| `hist3` | causal history L=3 + `[s_t^H; u_t]` | 60 | 0.898 |
| `hist6` | causal history L=6 + `[s_t^H; u_t]` | 120 | 0.919 |

Test = the held-out combined runs (unseen factor combinations) of the full
28-run dataset (20 train / 3 val / 5 test), with the zero-delta reference
recomputed on those exact test runs (`data/zero_baseline.json`). These are
the corrected full-set numbers; the earlier dev-set values are superseded
(see `sd_model/docs/stage19b_sd_variance_analysis.md`). The zero-delta
predictor has R2 = 0 by definition.

## Plots

```text
plots/loss_curves.png          train/val normalized loss vs epoch, all modes
plots/val_rmse_curves.png      val RMSE per channel group vs epoch
plots/test_rmse_comparison.png held-out test RMSE per group vs zero-delta
plots/r2_comparison.png        explained one-step variance per mode
plots/parity_hist3.png         predicted vs target scatter on test (best mode)
plots/rollout_rmse_vs_horizon.png  closed-loop trajectory error growth vs
                                   the persistence baseline
plots/per_domain_rmse.png      held-out one-step error per deployment domain
plots/summary_table.csv        machine-readable RMSE/R2 table
```

The rollout and per-domain plots come from
`sd_model/evaluation/evaluate_sd_rollout.py` (multi-step error growth with the
model's own state fed back, and error grouped by hydraulic-only /
mechanical-only / combined held-out runs). Rollout json files exist for all
five modes under `data/rollout_<mode>.json`, including `su_delta`.

## Regenerated artifacts (2026-08-08)

The plots, `data/zero_baseline.json`, `data/parity_hist3.npz` and
`plots/summary_table.csv` were regenerated from the full-set models so that
every number in this folder refers to the current checkpoints. Two code
fixes were required during the review:

- `sd_model/evaluation/evaluate_sd_rollout.py` now supports `--mode su_delta`
  and only evaluates history/delta modes from the trainer's first valid row
  per run (no invented warmup padding rows).
- `sd_model/make_plots.py` was missing `su_delta` from its mode list and its
  rollout plot passed per-horizon dicts to matplotlib; both are fixed.

## Reproduce

Train (needs the frozen + SD dataset, e.g. the Stage 19 dev set):

```text
python sd_model/training/train_sd_baseline.py --root <frozen_root> \
    --input su --epochs 150 --out outputs/sd_baseline/su
```

Regenerate the plots from the saved metrics + parity data:

```text
python sd_model/make_plots.py --root AAM_module/datasets/master_v1_frozen \
    --force-recompute
```

With `--root <frozen_root> --force-recompute` the parity data and zero-delta
reference are recomputed from the test runs instead of the stored files.

The stored `data/*.json` files in this folder were generated with exactly
that command against the full 28-run frozen dataset, so the checkpoints,
metrics, plots, zero baseline and summary table all refer to the same models.

Evaluate closed-loop behaviour on the held-out runs:

```text
python sd_model/evaluation/evaluate_sd_rollout.py --root <frozen_root> \
    --mode hist3 --checkpoint sd_model/checkpoints/hist3.pt \
    --out sd_model/data/rollout_hist3.json
```

## Source data

The full 28-run frozen dataset behind the current checkpoints and metrics:
20 train runs (nominal 4 / hydraulic 6 / mechanical 5 / combined 5),
3 validation runs (one hydraulic, one mechanical, one combined) and
5 held-out combined test runs with unseen factor combinations. Run-level
splits, train-only normalization, ~3768 aligned 50 ms transitions per run
(~105k total). Built by `AAM_module/tools/collect_run_structure.py` and the
Stage 16-18 pipeline, frozen at `AAM_module/datasets/master_v1_frozen`.
Full details and numbers: `sd_model/docs/stage19_sd_baseline.md` and
`sd_model/docs/stage19b_sd_variance_analysis.md`.

## Later stages in this folder

`stage20/` and `stage21/` hold the privileged-latent SD_z models and their
evaluation, following the same artifact layout (encoder/sd checkpoints,
`metrics.json`, `rollout.json`, `training.log`):

- `stage20/dz{2,4,8}` one-step joint System Factor Encoder sweep; `dz2` is
  the selected bottleneck (held-out test R2 0.785/0.793/0.795 for dz2/4/8).
- `stage20_unrolled/` RMA-style unrolled joint training (dz4).
- `stage21/dz2` combined one-step + rollout loss (ramp 150, early one-step
  phase) and `stage21/dz2_ramp60` (full curriculum to lambda_R=1.0; the
  reference Stage 21 model).
- `stage21/dz2_ramp60/rollout_latent_{true,zero,shuffled}.json` are the
  Stage 22 latent-matters ablation (true vs zero vs shuffled z).

Stage docs: `AAM_module/stage20_system_factor_encoder.md`,
`AAM_module/stage21_combined_rollout.md`, `AAM_module/stage22_latent_matters.md`.

## Stage 23 — frozen privileged representation

`stage23_frozen/dz2_nodp_holdq/` is the immutable frozen copy of the
selected Stage 22 model (encoder + SD + belief, `metrics.json` statistics,
explicit `protocol.json`, checksums and provenance). Created by
`sd_model/freezing/freeze_sd_model.py`; the frozen copy reproduces the Stage
22 rollout metrics exactly (verified by
`sd_model/freezing/verify_frozen_sd_model.py`, verdict under
`AAM_module/results/stage23_frozen_verify/`). The AAM stages consume this
copy as the stopped-gradient `z = mu(e)` source and must not fine-tune it.
