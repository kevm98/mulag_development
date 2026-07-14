# RRLAB MULAG hydraulic pipelines

This repository now contains two maintained workflows:

- `ID_FD_pipline/`: the deployed cylinder-space ID/FD actuator and residual PPO policy.
- `ROM-cylinders_pipeline/`: the hydraulic reduced-order cylinder model dataset, training,
  validation, and runtime pipeline.

Shared IsaacLab robot assets and task support remain under `extensions/`. The ID/FD runtime
dependencies, coordinate mapping, and original model architecture loaders are self-contained
under `ID_FD_pipline/runtime/` and `ID_FD_pipline/models/vendor_models/`.

## ID/FD residual policy

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/residual_policy/scripts/train.py \
  --headless --num_envs 64 --max_iterations 1000 \
  --curriculum_stage 1 --seed 17
```

See `ID_FD_pipline/README.md`, `ID_FD_pipline/TRAINING.md`, and
`ID_FD_pipline/residual_policy/README.md` for runtime and evaluation commands.

## Base cylinder ROM

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ROM-cylinders_pipeline/scripts/train_base_rom.py \
  ROM-cylinders_pipeline/data/base_rom_v1 \
  ROM-cylinders_pipeline/checkpoints/base_run_001 \
  --config ROM-cylinders_pipeline/configs/base_training.json
```

See `ROM-cylinders_pipeline/README.md` for dataset, validation, and runtime contracts.

TensorBoard logs remain under `logs/rsl_rl`.
