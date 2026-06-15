# Full Hydraulic Pipeline

Date: 2026-06-11

## High-Level Control

Task-space tracking uses PPO, not IK:

```text
Messerkopf ee_pose [x, y, z, qw, qx, qy, qz]
+ measured state
-> PPO Action[4]
```

The RRLAB obstacle task uses the same idea: `ee_pose` is an observation/reward command and PPO learns the mapping to arm action.

## Low-Level Learned Chain

```text
measured q[5], qdot[5], dP[4]
+ previous applied ValveCmd[4]
-> AAM history[100, 18]
-> frozen AAM from aam.pth
-> one stored z_arm_hat[24]
     |-> PPO observation indices 44:68
     `-> ID conditioning for the matching Action[4]

PPO Action[4]
-> qddot_cmd[5] in ID training units, qddot_cmd[4] = 0
-> ID conditioned by the stored z_arm_hat[24]
-> ValveCmd[4]
-> FD observer
-> valve-conditioned causal ROM, dt 0.05 s
-> DeltaExtension[4]
-> coordinate mapping
-> four prismatic position targets at Isaac dt 0.01 s
-> USD linkage
-> Messerkopf motion
```

The AAM runs online once per observed control state under inference mode. It is
frozen and in evaluation mode. `_prev_valve_cmd` is the previous ValveCmd that
was supplied to both FD and ROM, not a nominal or candidate command. AAM is
context only: it does not write ValveCmd, torque, ROM state, cylinder targets,
rewards, or task targets. The offline System Factor Encoder is not part of this
runtime chain.

The only actuator writer in ROM mode is:

```text
ROM -> robot.set_joint_position_target(... four cylinder joints ...)
```

## FD Observer

FD is active as:

```text
[q5, qdot5, dP4, ValveCmd4] history
-> FD
-> fd_tau_hat[4]
```

FD output is used for policy observation, load telemetry, safety fractions, and a small auxiliary reward. FD effort actuation is disabled in ROM position-target mode.

## Runtime Modes

- `joint_space_validation`: no checkpoint, stationary vehicle by default, scripted mechanical-joint reference, AAM/ID/ROM/FD observer active.
- `rl_task_space_tracking`: PPO training or checkpoint playback, RRLAB-style scripted vehicle motion, `ee_pose` command in observation/reward, PPO maps task-space state to `Action[4]`.

No mode claims task-space tracking without a trained PPO policy.

## Fixed Final Pose

The canonical `rl_task_space_tracking` experiment disables automatic curriculum
advancement and remains at Stage 1:

```text
episode length = 15 s
action authority = 0.25
vehicle motion = off
```

Each reset samples one fixed `Messerkopf` target in `robot_base` frame with
order `[x, y, z, qw, qx, qy, qz]`. `_ee_pose_command` is the only authoritative
target. It drives the 73D task-space observation, position error,
quaternion-geodesic orientation error, rewards, success logic, preflight, and
terminal metrics. Partial resets modify only selected target rows.

The policy still outputs `Action[4]`; no IK, trajectory optimizer, or direct
joint/cylinder command has been inserted. The learned and hydraulic chain below
the policy is unchanged.

Orientation reward is active with fixed weight `0.25` and tolerance
`0.0872664626 rad`. Position reward remains unchanged with weight `1.0`.

Success requires:

```text
position error <= 0.05 m
orientation error <= 0.0872664626 rad
measured EE linear speed <= 0.05 m/s
measured EE angular speed <= 0.10 rad/s
hold all conditions for 50 consecutive 0.01 s steps
```

Success terminates separately from unsafe conditions and timeout. It receives a
one-time bonus of `10.0` and no unsafe penalty. Terminal values are captured
before reset. These contracts have offline test coverage but still require
Isaac runtime validation.

**Contract assertions** (`_validate_control_mode()` for `rl_task_space_tracking`):

- 4 command joints required (cylinders)
- Observation dimension = 73 (`TASK_SPACE_OBSERVATION_DIM`)
- SD observer disabled (`use_sd_observer = False`)
- SD feedback disabled (`use_sd_feedback = False`)
- ID valve source active (`position_delta_source = "id_valve"`)
- FD effort actuation disabled (`use_fd_effort = False`)

## Tests

Focused offline contracts:

```bash
python kevin_integration/tasks/test_curriculum_learnability_repair.py
python kevin_integration/tasks/test_fixed_final_pose_task.py
python kevin_integration/tasks/test_aam_runtime_wiring_contract.py
```

The fixed-final-pose suite uses production Torch helpers and source inspection
to verify reset-only target sampling, quaternion geometry, settling success,
one-time bonus behavior, lifecycle ordering, fixed dimensions, frozen models,
follower ownership, and unchanged Hydra/launcher entrypoints.

The AAM contract suite loads production checkpoints and verifies the exact
18D feature order, 100-step history, finite 24D latent, shared observation/ID
latent, selected-row resets, fixed 73D observation and 4D outputs, and pure
history shapes for 1, 16, 600, and 2,000 environments. Isaac execution is still
required for live validation.
