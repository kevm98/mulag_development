# Hydraulic-aware cylinder ROM pipeline

This directory is isolated from `ID_FD_pipline`. Its current primary deliverable is a
valve-conditioned base hydraulic ROM:

```text
measured valve command + hydraulic/mechanical history
    -> pressure -> force/load -> velocity -> extension
```

The later command-conditioned PPO2 experiment implements:

```text
desired cylinder motion + measured hydraulic/mechanical history
    -> effective internal actuation
    -> pressure update
    -> hydraulic force and coupled load
    -> velocity update
    -> trapezoidal extension integration
    -> bounded PhysX cylinder position target
```

For that later experiment, the runtime command is the verified PPO1 pair
`[DesiredDeltaExtension(4), DesiredDeltaExtensionVelocity(4)]`. Actual valve commands are
optional training labels and are never runtime inputs.

## Important data status

The existing sibling directory is named `ROM-cylinders_pipeline ` with a trailing space. Its
`Models /ROM_cylinder_data.mat` is an AAM dataset: histories map to 91 constant randomized
plant parameters. The base dynamics ROM is therefore built directly from the four original
Simscape curriculum files, not from that derived MAT file.

## Base ROM dataset and training

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ROM-cylinders_pipeline/scripts/build_base_rom_dataset.py \
  ROM-cylinders_pipeline/data/base_rom_v1

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ROM-cylinders_pipeline/scripts/validate_base_rom_dataset.py \
  ROM-cylinders_pipeline/data/base_rom_v1 \
  --output ROM-cylinders_pipeline/reports/base_dataset_audit.json

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ROM-cylinders_pipeline/scripts/analyze_base_rom_delay.py \
  ROM-cylinders_pipeline/data/base_rom_v1 \
  --output ROM-cylinders_pipeline/reports/base_delay_analysis.json

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ROM-cylinders_pipeline/scripts/train_base_rom.py \
  ROM-cylinders_pipeline/data/base_rom_v1 \
  ROM-cylinders_pipeline/checkpoints/base_run_001 \
  --config ROM-cylinders_pipeline/configs/base_training.json
```

The base ROM uses the verified 0.05 s Simscape timestep and six samples of history (0.25 s).
It retains the actual valve as an explicit causal input. Valve-zeroed and valve-shuffled
validation must degrade before this model is accepted as hydraulically authoritative.
The logged `Force` signal fails the authored piston-area consistency check on the current
curriculum, so `base_training.json` assigns it zero supervised weight. The ROM still exposes
its pressure-derived hydraulic force; force supervision can be enabled after a reliable
chamber-force label is collected.

`scripts/run_base_rom_after_policy.sh` can wait for an existing training PID before starting
the GPU ROM training. It never signals or stops that process; it polls until the PID exits,
then trains and runs the offline acceptance gates.

## Dataset source contract

The flat source CSV must contain run/time/parameter identifiers, all 34 schema inputs, and the
four actual valve commands. See `ROM_pipeline/data/rom_schema.py` for exact names and ordering.
`intervention_id` is optional but should be supplied for matched-state command experiments.
Transitions and histories are built strictly within complete runs. Complete run/parameter
groups are assigned to train, validation, or test splits.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ROM-cylinders_pipeline/scripts/build_rom_dataset.py \
  path/to/closed_loop_rom_export.csv ROM-cylinders_pipeline/data/processed_v1

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ROM-cylinders_pipeline/scripts/validate_rom_dataset.py \
  ROM-cylinders_pipeline/data/processed_v1 \
  --output ROM-cylinders_pipeline/reports/dataset_audit.json
```

The legacy trailing-space dataset directory was audited and then removed during repository
cleanup. Its retained audit is `reports/legacy_mat_audit.json`.

## Training

Do not start this command while another GPU training job is active unless resources have been
deliberately allocated.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ROM-cylinders_pipeline/scripts/train_rom.py \
  ROM-cylinders_pipeline/data/processed_v1 \
  ROM-cylinders_pipeline/checkpoints/run_001 \
  --config ROM-cylinders_pipeline/configs/default_training.json
```

The trainer uses one-step hydraulic losses followed by recursive rollout horizons of 5, 20,
and 50 samples with scheduled sampling. The checkpoint embeds feature ordering, normalization,
timestep, model configuration, cylinder/joint ordering, and training provenance.

Train the required state-only comparison with the same configuration and `--state-only`, using
a separate output directory. This zeros the command throughout one-step and recursive training.

## Offline validation and tests

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ROM-cylinders_pipeline/scripts/validate_rom.py \
  ROM-cylinders_pipeline/checkpoints/run_001/best_rom.pt \
  ROM-cylinders_pipeline/data/processed_v1 \
  --output ROM-cylinders_pipeline/reports/offline_validation.json

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  ROM-cylinders_pipeline/tests -q
```

The current validation entry point reports grouped one-step scores, held-out recursive rollout
metrics, zeroed/shuffled-command ablations, the command-response Jacobian, condition number,
and inference time. Pressure-velocity phase-plane, frequency-response, and IsaacLab step gates remain
mandatory before PPO2 integration; one-step extension accuracy alone is not an acceptance
criterion.
