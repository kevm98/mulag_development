# ROM Validation Report

## Current Verdict

No ROM is selected yet, and no ROM is ready for Isaac integration.

This report covers methodology hardening plus smoke checks only. The 2-epoch GRU is not a serious candidate.

## Rollout Audit

- Audited checkpoint: `checkpoints/baselines/ridge_core_seed42/best_checkpoint.pth`
- Mode: `predicted_pressure`
- Feature index assertions: `Extension=8:12`, `dP=4:8`
- Update rule verified in report: raw `next = current + denormalized_delta`; normalization occurs only before model inference.
- Result: ridge catastrophic rollout is classified as **verified model instability or distribution-shift sensitivity under predicted-pressure feedback**, not an obvious raw-unit denormalization bug in the audited transitions.
- Audit summary over three validation runs: Extension full-run RMSE `231267136.88779816`, dP full-run RMSE `1.219139467930094e+17`, NaN/Inf count `0`, limit violations `2327`.

See `results/rollout_audit.md`.

## Locked Split Strategy

- Source dataset: `pytorch_hydraulic_rom_dataset.mat`
- Locked train: original train runs, 842 runs, 362902 samples
- Locked validation: original train runs, 94 runs, 40514 samples
- Locked test: original validation runs, 104 runs, 44824 samples
- Split seed: `42`
- Complete-run splitting verified: `True`
- Run overlap count: `0`
- Sequence-boundary violations: `0`

The locked test set exists but was not used for the smoke model-development checks in this report.

## Smoke Candidate Status

Best extension-only model: none selected.

The 2-epoch bounded GRU smoke checkpoint (`checkpoints/smoke_locked/gru_core_seed42/best_checkpoint.pth`) was used only to verify the locked extension-only workflow.

Locked validation one-step aggregate:

- DeltaExtension RMSE: `0.015528205782175064`
- DeltaExtension sign accuracy: `0.6626321690276943`
- R-squared: `0.0074220369264650365`

Per-cylinder sign accuracy:

- Cylinder 1: `0.7122633337458235`
- Cylinder 2: `0.7286982760752888`
- Cylinder 3: `0.6164427541242714`
- Cylinder 4: `0.5926756911724379`

This smoke GRU beats the zero baseline by a tiny margin on one-step RMSE but does not beat persistence (`0.007652221713215113`), so it fails selection gates.

Best extension+pressure model: none selected.

The existing ridge extension+pressure baseline has excellent one-step Extension RMSE but fails predicted-pressure rollout catastrophically.

## Rollout Status

Teacher-pressure rollout on three locked validation runs for the smoke GRU:

- 0.5 s Extension RMSE: `0.08109943173345746`
- 1.0 s Extension RMSE: `0.10461300567784745`
- full-run Extension RMSE: `0.7908743611811131`
- NaN/Inf count: `0`
- unstable rollout fraction: `0.0`
- training-range state-limit violations: `1` at 0.5 s, `11` at 1.0 s, `1196` full-run

Predicted-pressure rollout performance is not applicable to extension-only models. For the ridge extension+pressure baseline, predicted-pressure rollout is unstable by audit.

## Pressure And SD Decision

Pressure prediction is not stable enough for first deployment based on current evidence.

Recommended first practical option remains Option A:

`ROM predicts DeltaExtension only; SD/observer provides dP state.`

SD must remain out of the ROM training/control loop until units, target conventions, timestep, normalization, and dP channels are validated on synchronized held-out samples.

## Acceptance Blockers

- No serious 150-epoch locked-split GRU/LSTM/physics-informed training has been run in this session.
- No checkpoint beats persistence on locked validation/test one-step DeltaExtension RMSE.
- No checkpoint has passed locked test evaluation.
- No checkpoint has passed all-cylinder sign/magnitude acceptance.
- Teacher-pressure rollout still has training-range state-limit violations on the smoke GRU.
- Extension+pressure predicted-pressure rollout is unstable for ridge.
- Isaac integration must remain blocked until a core 26-input extension-only ROM passes the acceptance gates.

