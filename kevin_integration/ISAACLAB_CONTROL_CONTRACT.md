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

## Runtime Feature Order

AAM input:

```text
q[5], qdot[5], dP[4], prev_ValveCmd[4]
```

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

## dP Source

The active final ROM task uses:

```text
final_causal_rom_pressure_source = "training_mean"
```

Training-mean dP is read from the active ROM scaler before assembling the raw
26-feature ROM vector. SD feedback is disabled.

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

Reset telemetry logs `reset_count`, `reset_reason`, `episode_step`, `timeout`,
`unsafe_termination`, and `target_reached_termination`.

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
- verifies the task-space observation contract;
- verifies FD observer-only mode has no actuator conflict;
- verifies FD input/output contract readiness and finite `fd_tau_hat`;
- verifies wheel command resolution when scripted vehicle motion is enabled;
- reports `BLOCKED_FD_INPUT_CONTRACT`, `BLOCKED_FD_OUTPUT_NONFINITE`,
  `BLOCKED_FD_ACTUATOR_CONFLICT`, `BLOCKED_TASKSPACE_OBSERVATION`,
  `BLOCKED_TASKSPACE_REWARD`, `BLOCKED_VEHICLE_COMMAND`, or `READY_FOR_PPO`.

## Curriculum (Learnability Repair)

Four-stage curriculum active in `rl_task_space_tracking` mode:

| Stage | Steps         | Episode Length   | Action Authority | Vehicle    | Command Range         | Orientation |
|-------|---------------|------------------|------------------|------------|-----------------------|-------------|
| S1    | 0 – 50k       | 1500 st / 15 s   | 0.25×            | stationary | tight near nominal    | disabled    |
| S2    | 50k – 150k    | 1500 st / 15 s   | 0.50×            | stationary | moderate              | disabled    |
| S3    | 150k – 300k   | 3000 st / 30 s   | 0.75×            | stationary | wide                  | enabled     |
| S4    | 300k+         | 4500 st / 45 s   | 1.00×            | 3–8 km/h   | full                  | enabled     |

### Stage Detection

`_curriculum_stage()` compares `self.common_step_counter` (total env steps
across all environments) against `self.cfg.curriculum_stage_steps` (tuple of
upper bounds). Returns `1`, `2`, `3`, or `4`.

### Config Fields

All defined on `MulagAdaptiveRLJointReachEnvCfg`, overridden by
`MulagHydraulicROMTaskSpaceEnvCfg`:

| Field | Type | Description |
|-------|------|-------------|
| `curriculum_stage_steps` | `tuple[int, int, int]` | upper stage bounds `(50_000, 150_000, 300_000)` |
| `curriculum_episode_length_s` | `tuple[float, float, float, float]` | per-stage episode length in seconds |
| `curriculum_action_authority` | `tuple[float, float, float, float]` | per-stage action multiplier `(0.25, 0.50, 0.75, 1.00)` |
| `curriculum_vehicle_speed_min_kmh` | `tuple[float, float, float, float]` | per-stage min vehicle speed |
| `curriculum_vehicle_speed_max_kmh` | `tuple[float, float, float, float]` | per-stage max vehicle speed |
| `curriculum_vehicle_stationary_stages` | `int` | maximum stage index where vehicle is stationary (`3`) |
| `curriculum_command_x_range` | `tuple[tuple[float,float], ...]` | per-stage `[min, max]` x command range |
| `curriculum_command_y_range` | `tuple[tuple[float,float], ...]` | per-stage `[min, max]` y command range |
| `curriculum_command_z_range` | `tuple[tuple[float,float], ...]` | per-stage `[min, max]` z command range |
| `curriculum_orientation_enabled` | `tuple[bool, bool, bool, bool]` | per-stage orientation reward gate |

### Action Authority

In `_prepare_high_level_action()`, after action scaling:

```python
self._current_action = self._current_action * self._action_authority_multiplier
```

The multiplier is read from `curriculum_action_authority[stage-1]` during
`_update_curriculum_state()`. This reduces PPO action magnitude before qddot
conversion, preventing early-stage valve saturation and joint limit violations.

### Command Ranges

`_sample_task_space_command()` selects x/y/z ranges from stage-indexed
tuples. S1 x-range is `(-0.03, 0.03)`; S4 x-range is `(-0.30, 0.30)`. Z is
drawn as a constant from the range minimum.

Orientation quaternion:
- S1–S2: flat-facing `[0, 0, 1, 0]` (identity-like orientation, no orientation
  reward)
- S3–S4: identity `[1, 0, 0, 0]` (orientation reward active)

### Episode Length

`_get_dones()` computes stage-aware timeout:

```python
steps_per_episode = int(self.cfg.curriculum_episode_length_s[stage-1] / self.dt)
timeout = self.episode_length_buf >= steps_per_episode - 1
```

This gives shorter episodes in early stages (more resets per wall-clock hour
when the policy has not yet stabilised).

### Vehicle Speed

`_apply_rrlab_vehicle_command()` early-returns when
`stage <= curriculum_vehicle_stationary_stages` (S1–S3), zeroing both
`_vehicle_commanded_speed_kmh` and `_vehicle_wheel_velocity_target`. S4 samples
speed from `curriculum_vehicle_speed_min_kmh[3]` to
`curriculum_vehicle_speed_max_kmh[3]` (3–8 km/h).

## Termination Diagnostics

Per-episode booleans (reset on `_reset_idx()`):

```python
self._termination_joint_nonfinite[env_ids] = False
self._termination_lower_limit[env_ids] = False
self._termination_upper_limit[env_ids] = False
self._termination_velocity[env_ids] = False
self._termination_control_nonfinite[env_ids] = False
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

Task-space reward computed in `_get_task_space_rewards()`:

```python
pos_error = measured_ee_pos - commanded_ee_pos
pos_tracking = torch.exp(-4.0 * torch.norm(pos_error, dim=1))        # bounded [0,1]
progress = prev_pos_error_norm - current_pos_error_norm               # improving → positive
progress_clamped = progress.clamp(-rew_task_progress_clip, rew_task_progress_clip)
orient_enabled = curriculum_orientation_enabled[stage-1]
orient_reward = orient_enabled * rew_task_orient_tracking * torch.exp(-2.0 * orient_error)
```

Default weight values:

| Component | Weight |
|-----------|--------|
| `rew_task_pos_tracking` | 1.00 |
| `rew_task_orient_tracking` | 0.25 |
| `rew_task_progress` | 0.50 |
| `rew_task_progress_clip` | 0.20 |

Other penalties (action rate, action magnitude, valve saturation, clamp, FD
load, unsafe) are unchanged from the original `joint_reach_reward()`.
Episode sums for all 9 reward components are tracked and logged under
`extras["log"]["episode/..."]`.

## Contract Assertions

`_validate_control_mode()` for `rl_task_space_tracking` raises `RuntimeError`
if any contract is violated:

- `len(self._command_joint_ids) != 4` — must command exactly 4 cylinders
- Observation dimension != `TASK_SPACE_OBSERVATION_DIM` (73)
- `use_sd_observer = True` — SD observer must be disabled
- `use_sd_feedback = True` — SD feedback must be disabled
- `position_delta_source != "id_valve"` — must use ID valve source
- `use_fd_effort = True` — FD effort actuation must be disabled
