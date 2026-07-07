# Deprecated PPO1 Experiments

This archive preserves failed or superseded PPO1/new-observations experiments so
they are not accidentally used as the active controller path.

Do not import these folders for current PPO1 training:

- `experiments/ppo1_adapter/`: task-to-cylinder adapter experiment.
- `experiments/ppo1_nojac/`: direct cylinder-action PPO experiment.
- `experiments/ppo1_residual/`: residual Y/Z experiment.
- `experiments/pinned_static_jacobian/`: old pinned `J_pinv` wrappers.
- `scripts/ppo1_adapter/`, `scripts/ppo1_nojac/`, `scripts/ppo1_debug/`: matching scripts for archived experiments.
- `scripts/pinned_static_jacobian/`: scripts that import the pinned Y/Z wrappers.
- `scripts/legacy_workspace_diagnostics/`: one-off diagnostics that bypass the active PPO -> J_eff -> ID/FD path.
- `docs/`: previous notes that refer to superseded folder paths or abandoned experiments.

The active PPO1 controller remains:

```text
kevin_integration/new_observations_pipeline/ppo1/horizon_idfd_env_wrapper.py
```

with active scripts under:

```text
kevin_integration/new_observations_pipeline/scripts/ppo1/
```
