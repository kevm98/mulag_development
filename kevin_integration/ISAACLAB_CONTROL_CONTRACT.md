# IsaacLab Control Contract

## Action Meaning

Policy action is `Action[4]`, one normalized desired acceleration command per
hydraulic family:

```text
Action[4]
-> qddot_cmd[0:4] = Action[4] * ID_qddot_feature_std[0:4]
-> qddot_cmd[4] = 0
```

The qddot scale comes from the accepted ID checkpoint scaler entries for
`qddot1..qddot4`. This replaces the previous scalar `0.5`, which was far below
the ID training distribution, especially channels 3 and 4.

Current accepted ID qddot feature standard deviations:

```text
[4.087958, 2.653908, 15.052602, 19.057526]
```

Scripted demo qddot is built in raw physical units:

```text
qddot_des = kp_ref * (q_target - q) + kd_ref * (qdot_target - qdot) + qddot_feedforward
```

The feedback term, feedforward term, raw qddot, clipped action, and final
`qddot_cmd` are logged separately.

Task-space mode uses PPO as the bridge:

```text
Messerkopf ee_pose [x, y, z, qw, qx, qy, qz] in robot-base frame
-> PPO observation/reward
-> Action[4]
-> ID
```

The traced RRLAB obstacle task does not contain IK. It observes/rewards
`ee_pose`, and PPO learns the mapping to arm action.

## Local Launcher And Checkpoint Contract

Use the project-local player for Kevin task visualization:

```bash
./rrlab.sh -p kevin_integration/scripts/play_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 1 \
  --checkpoint /absolute/path/to/model_250.pt
```

The local player follows the installed IsaacLab RSL-RL player order, but adds
the Kevin task import after `AppLauncher` starts Isaac Sim and before
`@hydra_task_config` resolves the registry. It verifies `gym.spec()` and prints
an explicit diagnostic if the task is not registered. The generic IsaacLab
`play.py`, Hydra implementation, `rrlab.sh`, task name, and Gym task ID are not
modified.

Checkpoint compatibility checks compare available metadata for task name,
observation dimension, action dimension, runner class, policy architecture,
normalizer presence, environment contract version, and reward contract version.
Old checkpoints without Kevin metadata may be visualized only when tensor
dimensions remain compatible; they must not be used to resume training under a
changed reward contract without an explicit migration.

The deterministic evaluator writes:

```text
checkpoint_evaluation.json
checkpoint_evaluation.csv
best_checkpoint.json
```

Best-checkpoint selection is lexicographic: strict success rate, 50-step hold
completion fraction, mean maximum hold duration, final position error, final
orientation error, final linear speed, final angular speed, avoidable clamp
activity, and avoidable valve saturation. Mean reward is not a ranking key.

## Runtime Feature Order

AAM input:

```text
q[5], qdot[5], dP[4], previous applied ValveCmd[4]
```

The active AAM checkpoint is `kevin_integration/models/aam.pth`. It expects
history shape `(num_envs, 100, 18)` and returns `z_arm_hat[num_envs, 24]`.
Because `aam.pth` does not serialize its history length, runtime preserves the
existing production contract by using the accepted ID checkpoint's explicit
`window_size = 100`.

The complete adaptation branch is:

```text
q[5], qdot[5], dP[4], previous applied ValveCmd[4]
        |
        v
one authoritative AAM history
        |
        v
frozen online AAM
        |
        v
z_arm_hat[24]
      /             \
     v               v
PPO observation     ID conditioning
```

`_prev_valve_cmd` is the production field for previous applied ValveCmd. At
the end of each control step it is copied from `_current_valve_cmd`, the same
command already supplied to FD and the hydraulic ROM. AAM history is updated
once in `_get_observations`; ID is required to consume the stored latent and
cannot recompute AAM.

The task-space observation remains exactly:

```text
0:4    final target EE orientation quaternion
4:11   current EE pose
11:14  position error
14:15  orientation error
15:20  q[5]
20:25  qdot[5]
25:29  cylinder position[4]
29:33  cylinder velocity[4]
33:37  previous action[4]
37:41  previous applied ValveCmd[4]
41:65  z_arm_hat[24]
65:69  normalized FD observer output[4]
69:70  vehicle speed[1]
```

The AAM is frozen, in `eval()` mode, and invoked under
`torch.inference_mode()`. Shape and finiteness are checked before storage.
Per-environment version tensors verify that the latent inserted into a PPO
observation is the same stored latent supplied to ID for the matching action.
Partial reset clears only selected history, latent, and version rows.

AAM has no direct write path to ValveCmd, torque, ROM state, cylinder targets,
rewards, final-pose targets, or success. The System Factor Encoder is used
offline only.

ID input:

```text
q[5], qdot[5], qddot_cmd[5], dP[4]
```

FD observer input:

```text
q[5], qdot[5], dP[4], ValveCmd[4]
```

FD output:

```text
fd_tau_hat[4]
```

FD output is denormalized by `fd_scaler.pth` and used for observations,
load telemetry, safety fractions, and auxiliary reward. In ROM position-target
mode FD effort actuation is disabled.

Final ROM input, fixed 26-feature order:

```text
ValveCmd[4],
dP[4],
Extension[4],
ExtensionVelocity[4],
JointPos[5],
JointVel[5]
```

The final ROM output is:

```text
DeltaExtension_simscape[4] over 0.05 s
```

The active deployed ROM is valve-conditioned. Its dynamic correction explicitly
uses per-cylinder `ValveCmd`, `ValveCmd*abs(ValveCmd)`, `ValveCmd*dP_MPa`,
`dP_MPa`, `ExtensionVelocity`, centered `Extension`, and a small linear
JointPos/JointVel load correction. The previous causal linear export is kept
only as a transition-prediction baseline.

## Normalization

ID and SD scaler vectors may contain dynamic features followed by context
features. Runtime selects the first `dynamic_dim` entries only when checkpoint
metadata proves `dynamic_dim` equals the model input dimension. For ID this is
19 dynamic features.

AAM has no stored runtime normalizer in the checkpoint wrapper. Its checkpoint
lacks a stored window length, so runtime uses the ID checkpoint window as the
AAM fallback because ID is the only accepted checkpoint with an explicit
temporal contract.

Runtime telemetry is emitted through `extras["log"]`:

```text
adaptation/z_arm_hat_mean
adaptation/z_arm_hat_std
adaptation/z_arm_hat_abs_max
adaptation/z_arm_hat_finite_fraction
adaptation/history_abs_mean
adaptation/history_valve_abs_mean
adaptation/aam_active
adaptation/aam_frozen
```

These reductions remain on the active Torch device and are not appended to the
70D observation.

## Reward Contract

The current task-space reward contract is:

```text
kevin_fixed_pose_contract_version = kevin_fixed_pose_task_space_v2
reward_contract_version = task_space_reward_balance_v2
```

The environment prints the contract once at startup and writes
`reward_contract.json` into the run directory when `env_cfg.log_dir` is set.
The repaired terms use potential-based progress, bounded avoidable outward
clamp penalties, soft applied-ValveCmd saturation penalties, FD load excess
above the documented normalized threshold, near-pose speed penalties, settling
reward, hold-progress reward, and the existing one-time strict success bonus.
Reward-balance metrics are logged under `reward_balance/*` and do not enter the
observation.

## dP Source

The active final ROM task uses:

```text
final_causal_rom_pressure_source = "sd_observer"
```

The ROM pressure state is initialized from the active ROM scaler's training-mean
dP, then refreshed by the frozen SD observer for the raw 26-feature ROM vector.
SD feedback remains disabled; the SD output is used only as pressure
conditioning for the final causal ROM.

## Multirate Behavior

Isaac control step:

```text
0.01 s
```

Final ROM step:

```text
0.05 s
```

The multirate wrapper inserts a ROM history sample every fifth Isaac step,
predicts one `DeltaExtension_simscape[4]` when history is ready, and applies:

```text
DeltaExtension_simscape / 5
```

on each of the five 10 ms control steps.

## Coordinate Adapter

Active mapping:

```text
signs = [-1, -1, -1, -1]
x_sim = x_nominal - (q_isaac - q_isaac_nominal)
xdot_sim = -qdot_isaac
Delta_q_isaac = -Delta_x_sim
```

Offsets apply only to absolute position conversion. Velocity and increments use
sign only.

## Target Accumulation And Writer

Runtime accumulates Isaac prismatic targets:

```text
q_target_next = clamp(q_target_prev + Delta_q_isaac_10ms, soft_joint_limits)
```

The active writer is a single call per control step:

```text
robot.set_joint_position_target(q_target, joint_ids=command_joint_ids)
```

for:

```text
Drehzapfen_joint
Ausleger_I_joint
Ausleger_II_joint
Messerkopf_Schwenk_joint
```

## Reset Behavior

Reset clears cached deltas, pressure/Fnet state, previous actions, previous
valves, and initializes the ROM cylinder target from measured joint positions.
AAM/ID/FD/SD histories and the final ROM history are then warm-started from the
current measured state instead of left zero-filled.

Scripted demo behavior:

- episode length is 300 s;
- learned control is held off during the initial settle window;
- cylinder targets are re-seeded from measured post-settle positions;
- histories are filled from the settled state;
- learned control ramps in over 2 s;
- success/failure-style safety resets are suppressed in demo mode except for
  non-finite control values.

Reset telemetry distinguishes `final_pose_success`, `unsafe_termination`, and
`timeout`. Final pose, measured speed, hold counter, episode length, and
termination reason are snapshotted before physical reset.

Vehicle motion, when enabled, is scripted separately from PPO. The task writes
the same positive wheel angular target to the four wheel joints using the RRLAB
conversion `(speed_kmh / 3.6) / 0.575`, with speed sampled from 3 to 8 km/h.

## Preflights

Asset preflight:

- applies a short direct prismatic target pulse;
- checks measured cylinder motion;
- checks linked mechanical constraint-family motion;
- checks base z motion;
- uses a temporary direct-target writer only during the preflight.

Learned-pipeline preflight:

- runs the scripted reference through AAM -> ID -> final ROM -> coordinate
  mapping -> prismatic target accumulation;
- checks ValveCmd, ROM delta, target motion, measured cylinder motion, linkage
  motion, finite values, and clamp fraction.

Task-space bridge preflight:

- verifies the `Messerkopf` body resolves;
- verifies a finite task-space command/error tensor exists;
- verifies target and measured `[qw, qx, qy, qz]` quaternions are finite and
  have norm greater than `1e-8`;
- verifies the task-space observation contract;
- verifies FD observer-only mode has no actuator conflict;
- verifies FD input/output contract readiness and finite `fd_tau_hat`;
- verifies wheel command resolution when scripted vehicle motion is enabled;
- reports `BLOCKED_FD_INPUT_CONTRACT`, `BLOCKED_FD_OUTPUT_NONFINITE`,
  `BLOCKED_FD_ACTUATOR_CONFLICT`, `BLOCKED_TASKSPACE_OBSERVATION`,
  `BLOCKED_TASKSPACE_REWARD`, `BLOCKED_VEHICLE_COMMAND`, or `READY_FOR_PPO`.

## Fixed Final-Pose Task

The canonical task-space configuration uses:

```text
enable_automatic_curriculum = False
stage = 1
action authority = 0.25
episode length = 15 s
```

The existing four-stage implementation is retained, but `_update_curriculum_state()`
forces Stage 1 and `_evaluate_stage_advancement()` returns without advancing
while automatic progression is disabled.

`_ee_pose_command` is the single authoritative target:

```text
shape: (num_envs, 7)
order: [x, y, z, qw, qx, qy, qz]
frame: robot_base
```

`_sample_task_space_command(env_ids)` is called only by `_reset_idx(env_ids)`.
A partial reset samples only the selected rows and copies them to
`_final_pose_target_at_episode_start`. Observation and reward updates read this
target but never resample or govern it.

Measured EE pose is transformed from world to `robot_base` with
`subtract_frame_transforms`. Measured link linear and angular velocities are
converted to relative robot-base velocities before speed norms are computed.
IsaacLab quaternions remain in `[qw, qx, qy, qz]` order.

Target and measured quaternions are checked during preflight for finite values
and norm greater than `1e-8`, then normalized. Orientation error is the
sign-invariant quaternion geodesic:

```python
dot_abs = torch.abs(torch.sum(target_quat * current_quat, dim=-1)).clamp(0.0, 1.0)
orientation_error = 2.0 * torch.acos(dot_abs)
```

The dense unweighted orientation reward is:

```python
torch.exp(-torch.square(orientation_error / 0.0872664626))
```

Its fixed weight is `0.25`. Position tracking remains
`1.0 * exp(-4.0 * position_error_norm)`.

Final-pose success requires position error `<= 0.05 m`, orientation error
`<= 0.0872664626 rad`, measured EE linear speed `<= 0.05 m/s`, and measured EE
angular speed `<= 0.10 rad/s` for 50 consecutive control steps. Success is a
separate termination, produces a one-time `10.0` bonus, is excluded from timeout
and unsafe masks, and receives no unsafe penalty.

## Termination Diagnostics

Per-episode booleans (reset on `_reset_idx()`):

```python
self._termination_joint_nonfinite[env_ids] = False
self._termination_lower_limit[env_ids] = False
self._termination_upper_limit[env_ids] = False
self._termination_velocity[env_ids] = False
self._termination_control_nonfinite[env_ids] = False
self._termination_success[env_ids] = False
```

Offending joint/channel (reset to -1):

```python
self._termination_offending_joint[env_ids] = -1
self._termination_offending_channel[env_ids] = -1
```

Cross-episode accumulators (NOT reset — lifetime diagnostics):

```python
self._termination_count_joint_nonfinite += fired
self._termination_count_lower_limit += fired
self._termination_count_upper_limit += fired
self._termination_count_velocity += fired
self._termination_count_control_nonfinite += fired
self._termination_count_total += terminated
```

Logged to `extras["log"]` via `_populate_extras_log()`.

## Reward Composition

Task-space reward uses the authoritative target for both position and
quaternion-geodesic orientation tracking. Orientation tracking is active in the
fixed Stage 1 experiment.

Default weight values:

| Component | Weight |
|-----------|--------|
| `rew_task_pos_tracking` | 1.00 |
| `rew_task_orient_tracking` | 0.25 |
| `rew_task_progress` | 0.50 |
| `rew_task_progress_clip` | 0.20 |

Other penalties (action rate, action magnitude, valve saturation, clamp, FD
load, unsafe) are unchanged from the original `joint_reach_reward()`.
Terminal final-pose values are captured before `super()._reset_idx()` clears
episode state. Runtime metrics include target-change fraction, pose errors,
measured EE speeds, hold counter, success fraction, terminal success and
time-to-success, and separate accumulated success/velocity/timeout counts.
These metrics are not added to the PPO observation.

## Contract Assertions

`_validate_control_mode()` for `rl_task_space_tracking` raises `RuntimeError`
if any contract is violated:

- `len(self._command_joint_ids) != 4` — must command exactly 4 cylinders
- Observation dimension != `TASK_SPACE_OBSERVATION_DIM` (70)
- `use_sd_observer = True` — SD observer must be disabled
- `use_sd_feedback = True` — SD feedback must be disabled
- `position_delta_source != "id_valve"` — must use ID valve source
- `use_fd_effort = True` — FD effort actuation must be disabled
