# PPO1 New Observations Pipeline

This folder contains the active PPO1 controller cleanup for the Mulag arm. The
runtime path is intentionally narrow:

```text
PPO delta EE command
-> empirical J_eff
-> cylinder-space desired motion
-> ID/FD horizon rollout
-> cylinder targets
-> explicit_effort_pid_bias
-> PhysX
-> EE/tool motion and reward
```

Do not use `ppo1_adapter`, `ppo1_nojac`, pinned static `J_pinv`, or direct PPO
cylinder actions for current PPO1 work.

## Folder Structure

- `ppo1/`: active PPO1 environment, config, observations, rewards, ID/FD runtime
  adapters, and cylinder model utilities.
- `scripts/ppo1/`: active diagnostics and smoke tests for the PPO1 Horizon-IDFD
  path.
- `low_level_controller/`: support controllers used by `ppo1/`, including the
  empirical high-level EE controller and ID/FD rollout helpers.
- `learned_models/`: shared learned-model definitions and checkpoint notes.
- `tasks/`: task-space reference adapter glue.
- `docs/`: current generated diagrams.
- `archive/deprecated/`: preserved failed experiments and one-off diagnostics.

The teacher/student snapshot used as the organization reference keeps active
runtime code in clearly named package folders, scripts in a separate entry-point
tree, docs in docs-style locations, and old experiments out of the active import
path. This folder now follows that pattern.

## Active Vs Deprecated

Active runtime:

- `ppo1/horizon_idfd_env_wrapper.py`
- `ppo1/mulag_upper_level_env.py`
- `ppo1/mulag_upper_level_env_cfg.py`
- `ppo1/observations_adapter.py`
- `ppo1/rewards_adapter.py`
- `ppo1/cylinder_models_runtime.py`
- `ppo1/cylinder_space_scaling.py`
- `ppo1/joint_to_cylinder_adapter.py`
- `low_level_controller/high_level_ee_controller.py`
- `low_level_controller/idfd_rollout.py`

Active scripts:

- `scripts/ppo1/test_ppo1_idfd_smoke.py`
- `scripts/ppo1/debug_ground_clearance_horizon_idfd.py`
- `scripts/ppo1/sample_ground_workspace_horizon_idfd.py`
- `scripts/ppo1/compare_ppo_zero_random_horizon_idfd.py`

Deprecated but preserved:

- `archive/deprecated/experiments/ppo1_adapter/`
- `archive/deprecated/experiments/ppo1_nojac/`
- `archive/deprecated/experiments/ppo1_residual/`
- `archive/deprecated/experiments/pinned_static_jacobian/`
- `archive/deprecated/scripts/ppo1_adapter/`
- `archive/deprecated/scripts/ppo1_nojac/`
- `archive/deprecated/scripts/ppo1_debug/`
- `archive/deprecated/scripts/pinned_static_jacobian/`
- `archive/deprecated/scripts/legacy_workspace_diagnostics/`

## Data Flow

PPO produces a task-space delta-EE command in the base frame. The active wrapper
measures `J_eff` empirically by perturbing cylinder desired motion through the
same deployment chain used at runtime, then maps each PPO EE command to
cylinder-space desired motion with damped least squares. That desired motion is
rolled over the horizon with the learned ID/FD models, converted to cylinder
position targets, deployed with a smooth target ramp, and tracked by
`explicit_effort_pid_bias` in PhysX.

PPO observes task-space state only: EE pose, target pose/error, arm joint state,
previous high-level action, and base pose/velocity. Cylinder internals remain
inside the controller.

## ID/FD Usage

`HorizonIDFDEnv._idfd_rollout()` clones the live GRU histories, rolls the horizon
forward with:

- ID input: desired cylinder displacement/velocity plus current cylinder state.
- ID output: valve estimate.
- FD input/state: valve plus current cylinder extension/velocity history.
- FD output: predicted cylinder extension delta.

The rollout returns an accumulated `delta_fd_H`. Live ID/FD histories are
restored after planning, so planning does not mutate the simulator history.

## Empirical J_eff

In task-space mode, `HorizonIDFDEnv.reset()` measures `J_eff = d(EE_base) /
d(delta_x_des_H)` through the actual ID/FD and PhysX deployment chain. The
controller rejects non-finite or badly conditioned Jacobians in diagnostics and
prints rank/condition summaries. This replaces the old pinned static `J_pinv`
experiment, which is archived.

## Ground And Tool Contact

Ground trajectory code tracks the tool/contact bottom point, not just the EE
body origin. Contact position can be determined by:

- `geometry_aabb`: compute the world AABB for the configured tool body and use
  the bottom center.
- `manual_offset`: rotate a validated local tool-tip offset into world frame.
- `ee_origin`: retained only as an explicit warning mode.

The active diagnostics log:

- EE origin z
- tool/contact bottom z
- tool clearance
- arm-link clearance
- ground_z
- max_abs_body_world_position
- NaN/Inf count
- huge transform warning count

Hard ground safety failure is triggered for target clearance below the requested
tool clearance, tool sinking below the ground tolerance, arm geometry below the
ground tolerance, non-finite transforms, or huge transforms.

## Trajectory Visualization

`scripts/ppo1/debug_ground_clearance_horizon_idfd.py` can render:

- target path markers
- current target marker
- actual tool/EE trail
- tool contact marker
- optional tool AABB marker

Use a GUI Isaac Sim session for visual inspection. In headless mode the same
script still prints the safety metrics.

## Commands

Setup:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/rrlab_mulag
```

Headless smoke:

```bash
./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/ppo1/test_ppo1_idfd_smoke.py \
  --num_envs 1 --num_steps 3 --horizon_steps 2 --headless --device cuda:0
```

Ground trajectory debug:

```bash
./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/ppo1/debug_ground_clearance_horizon_idfd.py \
  --num_envs 1 --num_steps 40 --horizon_steps 2 --headless --device cuda:0 \
  --trajectory_mode ground_straight_line --visualize_target_trajectory false \
  --visualize_actual_ee_trail false
```

Zero/random comparison:

```bash
./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/ppo1/compare_ppo_zero_random_horizon_idfd.py \
  --num_envs 1 --num_steps 5 --horizon_steps 2 --headless --device cuda:0
```

PPO checkpoint comparison:

```bash
./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/ppo1/compare_ppo_zero_random_horizon_idfd.py \
  --num_envs 1 --num_steps 20 --horizon_steps 10 --headless --device cuda:0 \
  --checkpoint <PATH_TO_MODEL_PT>
```

Training:

```bash
./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 16 --max_iterations 200 --horizon_steps 10 \
  --horizon_action_scale_m 0.01 \
  --fd_horizon_gain_per_cylinder "2.0,3.0,3.0,3.0" \
  --headless --device cuda:0
```

## Control Diagram

- Source: `docs/control_pipeline.mmd`
- Image: `docs/control_pipeline.png`

## Validation Results

Last validation in this branch:

- `py_compile`: active PPO1/support/scripts compiled with
  `/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m py_compile`.
- script `--help`: active `scripts/ppo1/*` entry points returned help
  successfully.
- headless smoke:
  `test_ppo1_idfd_smoke.py --num_envs 1 --num_steps 1 --horizon_steps 1`
  completed one active ID/FD/PhysX step with no NaN observations or rewards.
- ground trajectory safety:
  `debug_ground_clearance_horizon_idfd.py --trajectory_mode ground_straight_line`
  reported `PASS: yes`, `safety_pass: yes`,
  `target_tool_bottom_clearance_min: 0.030000`,
  `tool_bottom_clearance_min: 1.578380`,
  `lowest_non_tool_arm_geometry_clearance_min: 2.025802`,
  `max_abs_body_world_position: 30.895414`, `NaN/Inf count: 0`, and
  `huge transform warning count: 0`.
- zero/random comparison:
  zero and random both had `nan_inf=0` and `huge_transform_warnings=0`.
  No active PPO checkpoint was supplied for a policy-vs-baseline comparison.
- PPO smoke:
  one PPO iteration completed with the active environment. It did not
  demonstrate learning: `success_rate: 0.0000`, `mean_final_ee_error: 0.3590`,
  `mean_reward: -131.25`.

Learning is not claimed. PPO must beat zero and random tracking with clean
physics counters before it is considered learning.

## Known Remaining Issues

- The active training entry point still lives in `kevin_integration/scripts/`
  for compatibility, but it instantiates `ppo1/HorizonIDFDEnv`.
- Ground trajectory completion depends on the sampled reachable workspace and
  the current calibrated PID bias.
- `geometry_aabb` requires USD geometry discovery; use `manual_offset` only
  after validating the offset.
- Current short runs still warn that empirical `J_eff` is finite but
  ill-conditioned, so task-space control quality remains a known issue.
