# Hydraulic Cylinder ROM Offline Workflow

This folder contains the offline workflow for reduced-order hydraulic cylinder
models. It intentionally does not modify the IsaacLab environment, USD, reward
terms, RL, AAM, ID, SD, or FD checkpoints.

No model is validated or selected until the acceptance gates at the bottom pass.

## Inspect And Audit

Inspect the MATLAB v7.3/HDF5 dataset:

```bash
python kevin_integration/rom_cylinders/inspect_rom_mat_dataset.py
```

Audit raw-unit autoregressive rollout before model selection:

```bash
python kevin_integration/rom_cylinders/audit_rom_rollout.py \
  --checkpoint kevin_integration/rom_cylinders/checkpoints/baselines/ridge_core_seed42/best_checkpoint.pth \
  --num_sequences 3 \
  --steps 20 \
  --rollout_mode predicted_pressure
```

The audit verifies `Extension=8:12`, `dP=4:8`, target denormalization, raw
state updates, and history updates. Output: `results/rollout_audit.md`.

## Locked Splits

Create locked complete-run train/val/test descriptors:

```bash
python kevin_integration/rom_cylinders/create_locked_rom_splits.py --seed 42
```

This splits original train runs into locked train/validation and reserves the
original validation split as locked test. Do not inspect locked test repeatedly
during model development.

Validate the loader:

```bash
python kevin_integration/rom_cylinders/test_rom_dataset.py --also_test_force --split_scheme locked
```

## Training

The default serious model target is extension-only:

```bash
python kevin_integration/rom_cylinders/train_rom.py \
  --model gru \
  --target_mode extension_only \
  --split_scheme locked \
  --epochs 150 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --hidden_dim 64 \
  --num_layers 2 \
  --dropout 0.1 \
  --early_stopping_patience 15 \
  --gradient_clip 1.0 \
  --bounded_outputs \
  --rollout_horizon 5 \
  --rollout_warmup_epochs 10 \
  --rollout_loss_weight 0.1
```

Also train locked-split LSTM and physics-informed extension-only candidates:

```bash
python kevin_integration/rom_cylinders/train_rom.py --model lstm --target_mode extension_only --split_scheme locked --epochs 150 --batch_size 256 --bounded_outputs --rollout_loss_weight 0.1
python kevin_integration/rom_cylinders/train_rom.py --model physics_informed --target_mode extension_only --split_scheme locked --epochs 150 --batch_size 256 --bounded_outputs --rollout_loss_weight 0.1
```

Keep extension+pressure as an experimental mode, not the default deployable
path:

```bash
python kevin_integration/rom_cylinders/train_rom.py --model gru --target_mode extension_and_dp --split_scheme locked --epochs 150 --batch_size 256 --bounded_outputs
```

Do not use Force by default. A Force-augmented model is only acceptable if an
equivalent online Force/Fnet signal is deployable and the model materially
improves held-out accuracy and rollout stability.

## Evaluation

One-step metrics:

```bash
python kevin_integration/rom_cylinders/evaluate_rom.py \
  --checkpoint <checkpoint> \
  --split val \
  --split_scheme locked
```

Teacher-pressure rollout for extension-only candidates:

```bash
python kevin_integration/rom_cylinders/rollout_rom.py \
  --checkpoint <checkpoint> \
  --split val \
  --split_scheme locked \
  --rollout_mode teacher_pressure
```

Full predicted-pressure rollout for extension+pressure candidates:

```bash
python kevin_integration/rom_cylinders/rollout_rom.py \
  --checkpoint <checkpoint> \
  --split val \
  --split_scheme locked \
  --rollout_mode predicted_pressure
```

Compare zero, persistence, and checkpoints under the selection rules:

```bash
python kevin_integration/rom_cylinders/compare_rom_models.py \
  --split val \
  --split_scheme locked \
  --target_mode extension_only \
  --include_rollout \
  --rollout_mode teacher_pressure \
  --checkpoint <checkpoint>
```

Outputs include:

- `results/model_comparison.csv`
- `results/model_comparison.md`
- `results/teacher_pressure_comparison.csv`
- `results/teacher_pressure_comparison.md`

After architecture and hyperparameters are selected, run the same evaluation on
`--split test --split_scheme locked` exactly once for final reporting.

## SD Compatibility

Document SD/ROM channel compatibility without wiring SD into ROM training:

```bash
python kevin_integration/rom_cylinders/evaluate_sd_rom_compatibility.py
```

Current first practical option: ROM predicts `DeltaExtension` only; SD or an
observer provides dP/Fnet state. Do not use SD as ROM feedback or correction
until passive compatibility is validated.

## Export And Inference

Export a selected checkpoint only after acceptance gates pass:

```bash
python kevin_integration/rom_cylinders/export_rom.py \
  --checkpoint <checkpoint> \
  --split val \
  --split_scheme locked \
  --include_rollout \
  --rollout_mode teacher_pressure
```

Inference parity:

```bash
python kevin_integration/rom_cylinders/test_rom_inference.py --export_dir kevin_integration/rom_cylinders/exported
```

Extension-only exports still return a zero-filled `DeltaDP` compatibility
channel, marked in debug as not model-predicted.

## Smoke Checks

Useful quick checks after code changes:

```bash
python -m py_compile kevin_integration/rom_cylinders/audit_rom_rollout.py \
  kevin_integration/rom_cylinders/create_locked_rom_splits.py \
  kevin_integration/rom_cylinders/train_rom.py \
  kevin_integration/rom_cylinders/evaluate_rom.py \
  kevin_integration/rom_cylinders/rollout_rom.py \
  kevin_integration/rom_cylinders/compare_rom_models.py \
  kevin_integration/rom_cylinders/export_rom.py \
  kevin_integration/rom_cylinders/rom_inference.py

python kevin_integration/rom_cylinders/train_rom.py \
  --model gru --target_mode extension_only --split_scheme locked \
  --epochs 2 --batch_size 64 --hidden_dim 16 --num_layers 1 --dropout 0 \
  --max_train_samples 1024 --max_val_samples 512 --bounded_outputs \
  --output_dir kevin_integration/rom_cylinders/checkpoints/smoke_locked
```

## Acceptance Gates

A model can only be selected if it satisfies all of these:

- beats zero and persistence on locked-test DeltaExtension RMSE
- beats persistence on teacher-pressure rollout at 0.5 s and 1.0 s
- has acceptable sign accuracy for every cylinder
- has no NaNs/Infs
- has low training-range state-limit violation rate
- passes inference parity
- does not depend on unavailable online Force
- performs adequately across randomized complete runs
- stays stable for all four cylinders

Only after those pass should Isaac integration be prepared, disabled by default,
as:

`ID valve command -> LearnedCylinderROM -> DeltaExtension -> accumulated cylinder position target -> prismatic cylinder position target`

`DeltaDP` remains auxiliary hydraulic-state output and must not be applied
directly to Isaac mechanical states.
