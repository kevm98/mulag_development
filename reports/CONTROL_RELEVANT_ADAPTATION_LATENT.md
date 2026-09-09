# Control-Relevant Privileged Adaptation Latent

## 1. Executive verdict

**NO-GO for a history → adaptation-latent AAM.** A jointly trained privileged encoder plus bounded residual adapter did not make correct plant context consistently more useful than mean or shuffled context on held-out runs. The ceiling test therefore stops before any online history estimator.

## 2. Why the previous seal-only route was abandoned

The earlier direct-factor work established that seal was observable, but the seal-conditioned oracle adapter was not better than mean/shuffled context. That route tested a physically named coordinate rather than a control-optimized representation and is closed. The old Stage-1 2-D dynamics latent and all historical artifacts remain preserved.

## 3. Existing rollout harness reused

| item | file/function |
|---|---|
| rollout evaluator | `AAM_module/tools/evaluate_stage29_latent_adapter.py:63-219` (same frozen-ID/FD mechanics reproduced for scalar-free privileged context evaluation) |
| frozen ID | `mulag_plant/runtime/cylinder_models_runtime.py:CylinderIDRuntime`; `learned_models/id_fd_model/best_id_cylinder_control.pt` |
| frozen FD | `CylinderFDRuntime`; `learned_models/id_fd_model/best_fd_cylinder_strict.pt` |
| target data | `AAM_module/datasets/master_v3_stage22_wide_frozen/raw` recorded cylinder state/reference transitions |
| context input | adapter feature row after 28 dynamic/base-ID features |
| horizons | H20 = 1 s, H100 = 5 s, H400 = 20 s at 20 Hz |
| metric | closed-loop cylinder tracking RMSE against desired reference |

## 4. New privileged adaptation encoder

`AAM_module/control_latent/train_privileged_adapter.py` adds `PrivilegedAdaptationEncoder`: normalized six-factor input → 32 → 16 → d=4. It is separate from the Stage-1 encoder and does not reconstruct physical factors.

## 5. Bounded adapter

The adapter is a 32-unit residual MLP conditioned on the 28 dynamic/base-ID features plus the learned context. Its output is `0.15*tanh(residual)` and is clipped by the frozen ID command limits. The frozen ID and FD are never updated.

## 6. Training objective

The encoder and adapter are trained jointly against the established frozen-FD next-state error, with a small normalized-residual penalty. Training used balanced deterministic samples from each train run, validation early stopping, one seed, and d=4 only.

## 7. Privileged vs mean/shuffled/base results

Reserved test runs: 16. Values are held-out closed-loop cylinder tracking RMSE (m).

| horizon | base | privileged | mean | shuffled |
|---|---:|---:|---:|---:|
| H20 / 1 s | 0.010550 | 0.010570 | 0.010570 | 0.010570 |
| H100 / 5 s | 0.010406 | 0.013553 | 0.013494 | 0.013577 |
| H400 / 20 s | 0.090309 | 0.109441 | 0.120271 | 0.090377 |

The critical inequalities do not hold consistently: privileged is worse than base at H20/H100, worse than mean at H100/H400, and worse than shuffled at H400. The small H400 shuffled advantage is not evidence of correct-context use.

## 8. Latent dimension comparison

Not run. d=4 failed the precondition for a dimension comparison; d=2 and d=8 would be post-hoc searching after a failed ceiling.

## 9. Paired held-out results

Per-run records are in `experiments/control_latent/d4_seed20260910/evaluation_h400.json`. The aggregate comparison is sufficient for the gate: privileged does not dominate both wrong-context controls at any reported horizon.

## 10. GO / NO-GO for history → z_adapt AAM

**NO-GO.** The current frozen dataset/model leaves no reliable context-specific privileged adaptation ceiling. Do not train `history → z_adapt`, do not tune latent dimensions, and do not proceed to E3/E5.

## 11. Exact files changed

- `AAM_module/control_latent/train_privileged_adapter.py`
- `AAM_module/control_latent/evaluate_privileged_adapter.py`
- `reports/CONTROL_RELEVANT_ADAPTATION_LATENT.md`

The obsolete AAM_v2/AAM-S Markdown reports created in the prior branch were deleted; datasets, checkpoints, JSON/NPZ results, plots, thesis source, PPO, ID, and FD were not deleted or retrained.

## 12. Exact next action

Stop this adaptation line; any future attempt needs a new control-relevant perturbation/data target or a changed correction interface before another online estimator is justified.
