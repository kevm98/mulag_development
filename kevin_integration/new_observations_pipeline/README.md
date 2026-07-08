# PPO1 New Observations Pipeline

This folder is the current PPO1 horizon-IDFD workspace for the MULAG arm. The
active controller remains under `ppo1/`; support code lives in the adjacent
package folders. Older experiments and stale wrappers were moved into
`backup_folder/` so the active path is easier to see without deleting anything.

Current active control path:

```text
PPO delta-EE command
-> empirical J_eff
-> cylinder-space desired motion
-> ID/FD horizon rollout
-> cylinder targets
-> PhysX implicit position drives
-> tool motion and reward
```

Do not use `ppo1_adapter`, `ppo1_nojac`, the pinned static `J_pinv` experiments,
or direct PPO cylinder actions for current PPO1 work.

## What To Open First

- `ppo1/README.md`: current operational notes, validation gates, training
  commands, live-plot command, and known limits.
- `ppo1/horizon_idfd_env_wrapper.py`: active horizon-IDFD wrapper and reward
  path.
- `ppo1/mulag_upper_level_env.py`: IsaacLab environment and actuation backend.
- `ppo1/mulag_upper_level_env_cfg.py`: active PPO1/controller configuration.
- `ppo1/ground_workspace_monte_carlo.py`: reachable/dynamic workspace sampling,
  observe mode, sweeps, validation summaries, and live plots.

## Folder Map

- `ppo1/`: active PPO1 runtime and current PPO1 tools. Core files stay here
  because training/play commands and imports expect this package path.
- `low_level_controller/`: empirical EE controller, horizon gain calibration,
  and ID/FD rollout helpers used by `ppo1/`.
- `learned_models/`: learned hydraulic model definitions and checkpoint notes.
- `tasks/`: task-space reference adapter glue.
- `docs/`: current diagrams, including the control-pipeline Mermaid source and
  rendered PNG.
- `backup_folder/`: preserved old/deprecated/generated files that are not part
  of the active PPO1 path.

## Active PPO1 Runtime Files

- `ppo1/horizon_idfd_env_wrapper.py`
- `ppo1/mulag_upper_level_env.py`
- `ppo1/mulag_upper_level_env_cfg.py`
- `ppo1/observations_adapter.py`
- `ppo1/rewards_adapter.py`
- `ppo1/cylinder_models_runtime.py`
- `ppo1/cylinder_space_scaling.py`
- `ppo1/joint_to_cylinder_adapter.py`
- `ppo1/sequence_cylinder_common.py`
- `low_level_controller/high_level_ee_controller.py`
- `low_level_controller/horizon_idfd_controller.py`
- `low_level_controller/idfd_rollout.py`

## Active PPO1 Tools

- `ppo1/ground_workspace_monte_carlo.py`: workspace sampling, observe mode,
  live plots, sweeps, and validation gates.
- `ppo1/collect_physx_id_dataset.py`
- `ppo1/collect_physx_fd_dataset.py`
- `ppo1/finetune_id_physx_dynamic.py`
- `ppo1/finetune_fd_physx_dynamic.py`

Training and playback entry points are outside this folder:

- `kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py`
- `kevin_integration/scripts/play_upper_level_ppo_horizon_idfd.py`

## Backup Folder

Nothing was deleted. Files moved out of the active view are kept under
`backup_folder/`:

- `backup_folder/deprecated_archive/`: the previous `archive/` tree, including
  failed/deprecated experiments such as `ppo1_adapter`, `ppo1_nojac`, residual
  PPO, and pinned-static-Jacobian work.
- `backup_folder/legacy_scripts/`: stale scripts/wrappers from the old
  `new_observations_pipeline/scripts/` tree.
- `backup_folder/legacy_ppo1_diagnostics/`: older PPO1 tuning and servo tests
  that are not the current validation path.
- `backup_folder/non_pipeline_ppo1/`: files moved out of `ppo1/` because they
  are not part of the new implicit-position horizon-IDFD pipeline.
- `backup_folder/generated_pycache/`: moved generated `__pycache__` folders.

See `backup_folder/README.md` for the backup inventory.

## Current Commands

Setup:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/rrlab_mulag
```

Headless zero-hold validation:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/ppo1/ground_workspace_monte_carlo.py \
  --mode observe \
  --workspace_json outputs/ppo1_horizon_idfd/dynamic_hold_workspace.json \
  --trajectory_mode ground_straight_line --trajectory_plane base_yz \
  --hold_target --observe_policy zero --observe_steps 60 \
  --actuator_backend implicit_position \
  --summary_json outputs/ppo1_horizon_idfd/observe_zero_hold_summary.json \
  --enforce_pass --headless --device cuda:0 --fast_exit
```

GUI observe run with live reference-vs-actual plots:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/new_observations_pipeline/ppo1/ground_workspace_monte_carlo.py \
  --mode observe \
  --workspace_json outputs/ppo1_horizon_idfd/dynamic_hold_workspace.json \
  --trajectory_mode ground_straight_line --trajectory_plane base_yz \
  --observe_policy greedy --greedy_gain 1.0 --observe_steps 400 \
  --actuator_backend implicit_position \
  --live_plots --device cuda:0
```

Training:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 16 --max_iterations 2000 \
  --trajectory_mode ground_straight_line --trajectory_plane base_yz \
  --workspace_json outputs/ppo1_horizon_idfd/dynamic_hold_workspace.json \
  --ground_max_episode_steps 300 \
  --headless --device cuda:0
```

## Diagrams

- Pipeline source: `docs/control_pipeline.mmd`
- Pipeline image: `docs/control_pipeline.png`

## Known Notes

- The active path uses the dynamic hold workspace, not the old kinematic-only
  workspace, when defining ground trajectories.
- Stage-0 local reach is not the validated plant path. Use ground trajectory
  tracking commands from `ppo1/README.md`.
- Keep new one-off diagnostics out of the active root unless they are part of the
  current validation workflow; put old experiments in `backup_folder/` instead.
