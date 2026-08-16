# Phase 1 factor-contract implementation — 2026-08-10

## Implemented

The shared verge task now has an opt-in, reset-latched deployment-factor
contract:

- `e = [flow, compliance, seal, pressure, mass_scale, com_z]` is selected per
  environment from the frozen Stage-22 manifest.
- Hydraulic values map to the command-ROM's fixed seven-channel factor order
  `[compliance, flow, pressure, seal, seal, seal, seal]`.
- `mass_scale` and `com_z` are applied to the PhysX `Messerkopf` body from an
  authored baseline on every reset, so mass and COM changes cannot compound.
- Domains are addressable as `nominal`, `hydraulic`, `mechanical`,
  `combined_train`, and `combined_test`. Explicit Gate-1 anchors include
  `flow_min`, `flow_nominal`, and `flow_max`.
- Factor keys and values are exposed as privileged episode metadata and logged
  at reset; they are not concatenated to policy observations or the causal AAM
  row. Construction checks preserve the existing policy width and 30-D AAM
  contract.
- The reserved 16 combined-test rows are loaded read-only and are not changed.

The default remains disabled to preserve historical nominal checkpoint
reproducibility. Evaluation enables it explicitly with `--factor_domain` or
`--factor_key`.

Example smoke commands:

```bash
python PPO_ROM/scripts/evaluate_gates.py --teacher --headless \
  --task RRLAB-Mulag-Verge-ROM-v0 --meshes 1337 --steps 4 --num_envs 1 \
  --factor_key flow_min

python PPO_ROM/scripts/evaluate_gates.py --teacher --headless \
  --task RRLAB-Mulag-Verge-ROM-v0 --meshes 1337 --steps 4 --num_envs 1 \
  --factor_key flow_max
```

Once simulator resources are available, the paired Gate-1 sweep is launched
with separate per-arm/per-key reports:

```bash
python PPO_ID_AAM_FD/scripts/evaluate_factor_sweep.py \
  --rom_checkpoint <PPO-ROM-checkpoint> \
  --idfd_checkpoint <PPO-ID-FD-checkpoint> \
  --out outputs/phase1_gate1_flow_sweep \
  --meshes 1337 --seed 17 --steps 960 --num_envs 16
```

## Verification

- `verge_task/tests/test_deployment_factors.py`: 4 tests passed.
- Python compilation and `git diff --check`: passed.
- Factor-enabled Isaac smoke reached and completed the short rollout without a
  factor-contract exception.

The simulator emitted repeated `ENOSPC` errors while creating Isaac file-watch
and GPU crash-dump artifacts during the second launch. The host filesystem has
free capacity, so this is an Isaac/runtime watcher-resource problem rather than
a factor or physics assertion. The full same-seed hydraulic sweep has not been
claimed or run yet.

## Gate 1 state

**Pending — do not proceed to Phase 1b or Stage 31.** The required three-point
same-seed/terrain sweep must still show monotonic PPO-ID-FD degradation while
PPO-ROM remains a matched-actuator reference. No ranges were widened and no
reserved test configuration was altered.
