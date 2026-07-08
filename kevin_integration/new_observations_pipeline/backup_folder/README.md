# Backup Folder

This folder holds files moved out of the active
`kevin_integration/new_observations_pipeline/` view. They are preserved for
reference, but they are not part of the current PPO1 horizon-IDFD control path.

## Contents

- `deprecated_archive/`: the old `archive/` tree. This includes failed or
  superseded experiments such as `ppo1_adapter`, `ppo1_nojac`, residual PPO, and
  pinned-static-Jacobian work.
- `legacy_scripts/`: the old `new_observations_pipeline/scripts/` tree. Current
  PPO1 validation and observe commands now use scripts directly under `ppo1/` or
  the training/play entry points under `kevin_integration/scripts/`.
- `legacy_ppo1_diagnostics/`: older PPO1 tuning scripts that were useful during
  debugging but are not the active validation path.
- `non_pipeline_ppo1/`: files that were still sitting in `ppo1/` but belonged to
  older/non-active paths, including the direct-cylinder action adapter, the old
  explicit-backend root-cause harness, and the older deployment actuator sweep.
- `generated_pycache/`: generated Python bytecode cache folders moved out of the
  active package view.

No important files were deleted during the cleanup.
