# Kevin Integration Codebase Inspection Report

**Date:** 2026-06-11  
**Scope:** `kevin_integration/` on the `integration/taskspace-fd-hydraulic-pipeline` branch  
**Method:** Static source-code analysis via file reads (no execution, no git CLI, no Conda)

---

## Section A: Repository State

> UNCERTAIN — requires shell access (not available in architect mode).  
> Cannot confirm current branch, dirty/untracked files, or diff with `main`.  
> The file tree shows only the `kevin_integration/` subtree. It is assumed the reviewer is on the correct branch.

---

## Section B: Task Registration

**File:** [`kevin_integration/tasks/__init__.py`](../kevin_integration/tasks/__init__.py)

Four gym registrations exist:

| Task ID | Config | `control_mode` | Purpose |
|---------|--------|----------------|---------|
| `Kevin-Mulag-HydraulicROM-TaskSpace-v0` | [`MulagHydraulicROMTaskSpaceEnvCfg`](../kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py:310) | `rl_task_space_tracking` | Primary task-space PPO |
| `Kevin-Mulag-AdaptiveRL-JointReach-v0` | `MulagHydraulicROMTaskSpaceEnvCfg` *(same)* | `rl_task_space_tracking` | Alias — identical config |
| `Kevin-Mulag-AdaptiveRL-v0` | `MulagHydraulicROMTaskSpaceEnvCfg` *(same)* | `rl_task_space_tracking` | Alias — identical config |
| `Kevin-Mulag-HydraulicROM-JointTracking-v0` | [`MulagHydraulicROMJointTrackingEnvCfg`](../kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py:274) | `joint_space_validation` | Validation-only, stationary vehicle |

**Key observation:** Three aliases (`TaskSpace-v0`, `AdaptiveRL-JointReach-v0`, `AdaptiveRL-v0`) all resolve to the same `MulagHydraulicROMTaskSpaceEnvCfg`. This is correct for task-space tracking but means the aliases provide no differentiated behaviour. The aliasing is intentional for backward compatibility / legacy naming.

---

## Section C: Observation Contracts — 62D vs 73D

Two distinct observation builders exist in [`kevin_integration/rl/observation_builder.py`](../kevin_integration/rl/observation_builder.py):

### 62D — [`build_rl_obs()`](../kevin_integration/rl/observation_builder.py:35) (joint-space validation mode)

| Slice | Fields | Dim |
|-------|--------|-----|
| 0:5 | `q` (joint position) | 5 |
| 5:10 | `qdot` (joint velocity) | 5 |
| 10:15 | `q_ref` (reference position) | 5 |
| 15:20 | `qdot_ref` (reference velocity) | 5 |
| 20:25 | `q_error` = `q_ref - q` | 5 |
| 25:30 | `qdot_error` = `qdot_ref - qdot` | 5 |
| 30:34 | `prev_action` (PPO output) | 4 |
| 34:38 | `prev_valve_cmd` (from ID) | 4 |
| 38:62 | `z_arm_hat` (from AAM) | 24 |
| **Total** | | **62** |

### 73D — [`build_task_space_rl_obs()`](../kevin_integration/rl/observation_builder.py:73) (task-space PPO mode)

| Slice | Fields | Dim |
|-------|--------|-----|
| 0:7 | `ee_pose_command` (desired Messerkopf pose) | 7 |
| 7:14 | `current_ee_pose` (current Messerkopf pose) | 7 |
| 14:17 | `ee_position_error` | 3 |
| 17:18 | `ee_orientation_error` (1D angle) | 1 |
| 18:23 | `mechanical_q` (joint position, 5-DOF) | 5 |
| 23:28 | `mechanical_qdot` | 5 |
| 28:32 | `cylinder_q` (prismatic position, 4) | 4 |
| 32:36 | `cylinder_qdot` | 4 |
| 36:40 | `prev_action` | 4 |
| 40:44 | `prev_valve_cmd` | 4 |
| 44:68 | `z_arm_hat` | 24 |
| 68:72 | `fd_tau_hat` | 4 |
| 72:73 | `vehicle_speed_kmh` | 1 |
| **Total** | | **73** |

**Constants defined in:** [`kevin_integration/tasks/task_space_reference.py`](../kevin_integration/tasks/task_space_reference.py)  
`TASK_SPACE_OBSERVATION_DIM = 73` at line 7.

**Selection logic** in [`_get_observations()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1562):

```python
if self._task_control_mode() == "rl_task_space_tracking":
    obs = build_task_space_rl_obs(...)  # 73D
    expected_obs = int(getattr(self.cfg, "task_space_observation_space", TASK_SPACE_OBSERVATION_DIM))
else:
    obs = build_rl_obs(...)  # 62D
    expected_obs = int(self.cfg.observation_space)
```

The branching condition is `self._task_control_mode()` which returns `self.cfg.control_mode` if it is a recognised value, else `self.cfg.task_control_mode` (line 397-399).

---

## Section D: Current Preflight Failure Trace

**Confirmed bug:** [`run_learned_pipeline_preflight()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:2778) at line 2793-2795.

### The mutation

```python
# Line 2793-2795:
original_mode = self.cfg.control_mode
self.cfg.control_mode = "scripted_reference_demo"   # MUTATION
self.reset()                                          # Triggers _get_observations()
```

### The crash sequence

1. `train_mulag_adaptive_rl.py` → [`_run_motion_preflights()`](../kevin_integration/scripts/train_mulag_adaptive_rl.py:345) → calls `env.run_learned_pipeline_preflight()`
2. Preflight mutates `self.cfg.control_mode = "scripted_reference_demo"`
3. Preflight calls `self.reset()` (line 2795)
4. `reset()` → `DirectRLEnv.reset()` → `_reset_idx()` → ... eventually `_get_observations()` is called
5. `_task_control_mode()` returns `"scripted_reference_demo"` — which is **not** `"rl_task_space_tracking"`
6. Therefore the `else` branch is taken: 62D observation via `build_rl_obs()`
7. But the environment was created with `observation_space = 73` (from `MulagHydraulicROMTaskSpaceEnvCfg`)
8. **Crash:** `"Observation size mismatch: got 62, expected 73"` (line 1585)

### Root cause

`_task_control_mode()` (line 397) checks if `self.cfg.control_mode` is in the set `{"rl_task_space_tracking", "joint_space_validation", "scripted_reference_demo"}`. When set to `"scripted_reference_demo"`, it returns that value literally. The `_get_observations()` switch then sees a non-task-space mode and selects the 62D path.

The preflight temporarily needs `"scripted_reference_demo"` behaviour (scripted joint references during warm-up), but after `reset()` returns, the mode is restored at line 2802-2803:

```python
self.cfg.control_mode = original_mode
```

**The problem:** `reset()` calls `_get_observations()` internally (through the IsaacLab `DirectRLEnv` lifecycle), and that call fails because the observation dimension was already locked to 73 at construction time.

### Fix direction

The fix must ensure that during preflight mode-switch, either:
- The observation dimension check is bypassed, OR
- The `_get_observations()` switch condition includes `"scripted_reference_demo"` as an alias for `"rl_task_space_tracking"`, OR
- The preflight builds 62D observations temporarily and the environment accepts the dimension change

---

## Section E: Actuation Writer Audit

**File:** [`kevin_integration/tasks/mulag_adaptive_rl_env.py`](../kevin_integration/tasks/mulag_adaptive_rl_env.py)

### Writer exclusivity enforcement

[`_assert_single_actuator_writer()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:353) checks that IsaacLab's `actuator_net` and the custom ROM writer are not both active. This prevents conflicting writes.

### [`_apply_action()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1525) dispatch logic

```python
if self.cfg.use_final_causal_cylinder_rom:
    # ROM mode: position target via set_joint_position_target (line 1533)
    for joint_name, target_pos in zip(self._rom_joint_names, isaac_delta):
        self._robot.set_joint_position_target(target_pos, joint_ids=[...])
else:
    # Legacy FD effort path: set_joint_effort_target (line 1557)
    effort_5d = torque_to_joint_effort(fd_tau)
    self._robot.set_joint_effort_target(effort_5d, joint_ids=self._robot_joint_ids)
```

**ROM path:** Uses [`set_joint_position_target`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1533) — writes position targets to the 4 ROM cylinder prismatic joints + Messerkopf_joint (joint 5, which follows via coupling).

**FD effort path:** Uses [`set_joint_effort_target`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1557) — writes 5-DOF effort to all arm joints.

**Vehicle actuation** is separate: [`_apply_rrlab_vehicle_command()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:930) uses `write_joint_velocity_to_sim` for the 4 wheel joints — independent of the arm actuator writer.

**Conclusion:** The actuator writer exclusivity is correctly enforced. ROM position-target and FD effort paths are mutually exclusive via both the high-level `_assert_single_actuator_writer()` and the `_apply_action()` dispatch.

---

## Section F: ROM Runtime Audit

### Valve-conditioned causal ROM

**Model:** [`ValveConditionedCausalCylinderROM`](../kevin_integration/rom_cylinders/valve_conditioned_causal_rom.py:40)  
**Runtime wrapper:** [`ValveConditionedROMInference`](../kevin_integration/rom_cylinders/valve_conditioned_causal_rom.py:154)  
**Multirate wrapper:** [`MultirateROMWrapper`](../kevin_integration/rom_cylinders/multirate_rom_wrapper.py:31)

### Architecture

- **ROM dt:** 0.05 s (20 Hz)
- **Isaac control dt:** 0.01 s (100 Hz)
- **Ratio:** 5 (each ROM step spans 5 Isaac steps)
- **Input:** 26-dim feature vector per timestep: [ValveCmd(4), dP(4), Extension(4), ExtensionVelocity(4), JointPos(5), JointVel(5)]
- **History:** 10 timesteps at 0.05 s spacing = 0.5 s window
- **Output:** `final_delta_extension` (4) = kinematic baseline + bounded tanh correction

### Feature computation (17 coefficients per cylinder)

[`_final_causal_rom_feature_row()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1376) assembles:
```
features = [1.0, u, u*|u|, u*dP, dP, v, x, q1..q5, qv1..qv5]
```
where `u` = valve command, `dP` = pressure delta, `v` = extension velocity, `x` = extension, `q` = joint pos, `qv` = joint vel.

### ROM metadata

From [`best_valve_conditioned_causal_rom_metadata.json`](../kevin_integration/rom_cylinders/exported_valve_conditioned_causal_rom/best_valve_conditioned_causal_rom_metadata.json):

| Metric | Value |
|--------|-------|
| Model type | "Model D valve-pressure-load structured regression" |
| Parameter count | 68 (17 features × 4 cylinders) |
| Correction bound (per cyl) | [0.019, 0.015, 0.078, 0.071] m |
| One-step RMSE (valve-cond) | [0.0013, 0.0010, 0.0056, 0.0048] m |
| One-step R² (valve-cond) | [0.961, 0.982, 0.947, 0.918] |
| Rollout RMSE @ 0.1s | 0.0053 m |
| Rollout RMSE @ 2.0s | 0.0388 m |
| Channel isolation | PASS (off-diag ratios > 3×10⁸) |

### Cylinder coordinate mapping

**File:** [`cylinder_coordinate_adapter.py`](../kevin_integration/rom_cylinders/cylinder_coordinate_adapter.py)  
**Mode:** `nominal_sign_offset` (sign + offset from nominal joint/cylinder detent pose)  
**Mapping JSON:** [`cylinder_coordinate_mapping.json`](../kevin_integration/rom_cylinders/cylinder_coordinate_mapping.json)

The mapping converts between Simscape extension space (ROM output) and Isaac prismatic position space:

```
isaac_pos = isaac_nominal + sign * (simscape_ext - simscape_nominal)
```

Four cylinders in order: `Drehzapfen_joint`, `Ausleger_I_joint`, `Ausleger_II_joint`, `Messerkopf_Schwenk_joint`

### ROM pipeline execution

In [`_pre_physics_step_final_causal_rom()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1181):

1. Read dP (pressure deltas) — lines 1192-1197
2. Read cylinder state (mapped from mechanical) — line 1200
3. Read mechanical joint state — line 1203
4. Run learned controller to get ValveCmd — line 1206
5. Run FD observer-only (zero torque output) — line 1209
6. Assemble ROM feature row — lines 1211-1246
7. Step the multirate ROM wrapper — lines 1250-1253
8. Convert Simscape delta to Isaac delta — line 1257
9. Apply position target — lines 1259-1286

---

## Section G: FD Runtime Audit

### FD is observer-only in ROM mode

[`_run_fd_observer_only()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1034) explicitly zeros the FD torque output:

```python
def _run_fd_observer_only(self, q, qdot, dP, valve_cmd):
    fd_x = build_fd_input(q, qdot, dP, valve_cmd)
    fd_tau_hat = self._controller.run_fd_observer(fd_x)
    # NOT applied to simulation
    # Only stored for observation: fd_tau_hat goes into 73D obs
```

This is confirmed in the ROM pipeline at line 1209:
```python
self._run_fd_observer_only(q, qdot, dP, valve_cmd)
```

### When is FD active (non-ROM mode)?

In the legacy path ([`_pre_physics_step_reduced_hydraulic_rom()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1420)), when `use_final_causal_cylinder_rom=False`:
- FD output is adapted via `adapt_fd_to_isaac_torque()` in [`torque_adapter.py`](../kevin_integration/rl/torque_adapter.py)
- Applied as joint effort targets to the simulation

### FD model

**Class:** [`ForwardDynamicsModel`](../kevin_integration/controllers/learned_models.py:554)  
**Type:** LSTM-based, learned from Simscape data  
**Input (18-dim):** [q(5), qdot(5), dP(4), valve_cmd(4)]  
**Output (4-dim):** Simscape torque estimate per cylinder  
**Scaler:** Loaded from `fd_scaler.pth` in [`LearnedControlPipeline`](../kevin_integration/controllers/pipeline.py:65)

### Torque adapter modes

From [`torque_adapter.py`](../kevin_integration/rl/torque_adapter.py):
- `"none"`: raw FD output
- `"scale_bias"`: `scale * fd_output + bias`
- `"tanh_squash"`: `max_abs * tanh(fd_output / fd_scale)`
- `"residual_pd"`: PD torque + scaled FD residual

Default preset: `"conservative"` with scale `[0.0001, 0.001, 0.0005, 0.0001]`

---

## Section H: Mechanical vs Cylinder State

### Mechanical joint space (5-DOF)

Order: `Drehzapfen_joint`, `Ausleger_I_joint`, `Ausleger_II_joint`, `Messerkopf_Schwenk_joint`, `Messerkopf_joint`

### Cylinder/prismatic space (4-DOF)

Order: `Drehzapfen_joint`, `Ausleger_I_joint`, `Ausleger_II_joint`, `Messerkopf_Schwenk_joint`  
(Messerkopf_joint has no prismatic cylinder — it is a rotational joint coupled to Messerkopf_Schwenk_joint via `joint5_coupling`.)

### Mapping

[`_mapped_cylinder_state()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1354):

```python
def _mapped_cylinder_state(self):
    cyl_q = read_prismatic_position(4 cylinders)
    cyl_qdot = read_prismatic_velocity(4 cylinders)
    # Map via CylinderCoordinateMapping
    mapped_ext = self._cyl_coord_mapping.isaac_position_to_simscape_extension(cyl_q)
    mapped_vel = self._cyl_coord_mapping.isaac_velocity_to_simscape_velocity(cyl_qdot)
    return mapped_ext, mapped_vel
```

The mapping converts from Isaac USD prismatic positions to Simscape extension coordinates using `CylinderCoordinateMapping` with nominal sign/offset.

### State names

From [`rrlab_reference_conventions.py`](../kevin_integration/tasks/rrlab_reference_conventions.py:6):
```python
OBSTACLE_ARM_JOINT_NAMES = (
    "Drehzapfen_joint",        # index 0
    "Ausleger_I_joint",        # index 1
    "Ausleger_II_joint",       # index 2
    "Messerkopf_Schwenk_joint",# index 3
    "Messerkopf_joint",        # index 4
)
```

Cylinder order from [`cylinder_coordinate_adapter.py`](../kevin_integration/rom_cylinders/cylinder_coordinate_adapter.py:11):
```python
CYLINDER_ORDER = (
    "Drehzapfen_joint",        # cylinder 0 (rom_index 0)
    "Ausleger_I_joint",        # cylinder 1 (rom_index 1)
    "Ausleger_II_joint",       # cylinder 2 (rom_index 2)
    "Messerkopf_Schwenk_joint",# cylinder 3 (rom_index 3)
)
```

---

## Section I: Task-Space Control Audit

### Intended architecture (ROM mode)

```
PPO policy → Action[4] → AAM → z_arm_hat[24] → ID → ValveCmd[4]
    → Valve-conditioned causal ROM → DeltaExtension[4]
    → CylinderCoordinateMapping → Isaac position target[4]
    → set_joint_position_target → cylinder/linkage → Messerkopf motion
```

### Action pipeline (non-ROM / legacy)

```
PPO policy → Action[4] → AAM → z_arm_hat[24] → ID → ValveCmd[4]
    → FD → fd_tau[4] → TorqueAdapter → joint effort[5]
    → set_joint_effort_target → arm motion
```

### Scripted reference mode

The preflight and scripted reference paths use [`_scripted_reference_action()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1010) which computes:

```python
action = (q_ref - q) * Kp + (qdot_ref - qdot) * Kd
```

This is a PD tracking controller that generates actions from reference joint trajectories.

### Task-space bridge

[`run_task_space_bridge_preflight()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:527) validates the end-effector pose tracking pipeline. It:
1. Opens the ROM-to-Isaac adapter
2. Runs a sample pose command through the pipeline
3. Verifies the Messerkopf body responds correctly

---

## Section J: Vehicle Control Audit

### Reference conventions

From [`rrlab_reference_conventions.py`](../kevin_integration/tasks/rrlab_reference_conventions.py):

| Parameter | Value |
|-----------|-------|
| Wheel radius | 0.575 m |
| Speed range | 3.0 — 8.0 km/h |
| Speed conversion | `kmh_to_wheel_rad_per_sec(speed_kmh, wheel_radius)` |

Wheel joint names:
```python
("Wheel_Front_Left_joint", "Wheel_Front_Right_joint",
 "Wheel_Rear_Left_joint", "Wheel_Rear_Right_joint")
```

### Implementation in env

[`_apply_rrlab_vehicle_command()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:930):

```python
def _apply_rrlab_vehicle_command(self):
    speed_kmh = self._scripted_vehicle_speed  # sampled near reset
    omega = kmh_to_wheel_rad_per_sec(speed_kmh, OBSTACLE_WHEEL_RADIUS_M)
    self._robot.write_joint_velocity_to_sim(
        torch.full((self.num_envs, 4), omega, device=self.device),
        joint_ids=self._wheel_joint_ids
    )
```

### RRLAB reference task

The reference [`RRLAB-Obstacle-Avoidance-Mulag-v0`](../extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/config/mulag/joint_pos_env_cfg.py) uses:
- Same wheel joints
- Driving speed randomisation via `random_driving_speed` at (3.0, 8.0) km/h
- Wheel velocity range (3.0, 8.0) km/h → rad/s conversion

**Conclusion:** The vehicle control in `kevin_integration` matches the RRLAB obstacle avoidance reference conventions exactly.

---

## Section K: Legacy and Conflicting Paths

### What is legacy (not used in ROM mode)

1. **[`_pre_physics_step_reduced_hydraulic_rom()`](../kevin_integration/tasks/mulag_adaptive_rl_env.py:1420)** — Older reduced hydraulic actuator ROM path. Uses `ReducedHydraulicActuatorROM` (conservative bootstrap parameters in [`reduced_hydraulic_actuator.py`](../kevin_integration/controllers/reduced_hydraulic_actuator.py)). Only active when `use_final_causal_cylinder_rom=False`.

2. **[`apply_action_full_pipeline()`](../kevin_integration/rl/adaptive_rl_controller.py:458)** — Full AAM → ID → FD → torque path. Writes FD-estimated torques as joint efforts. Not called in ROM mode (only `apply_action_valve_only()` is used).

3. **[`LearnedControlPipeline`](../kevin_integration/controllers/pipeline.py)** — Older stand-alone pipeline (not env-integrated). Uses numpy arrays, separate `SequenceBuffer` objects. Runs AAM → ID → FD sequentially as one call. **Not wired into the main env flow** — the env uses `AdaptiveRLLearnedController` instead.

### Potential conflicts

- The `_pre_physics_step()` dispatch (line 1064) checks `use_final_causal_cylinder_rom` first. If True → ROM path. If False → checks various other flags. The dispatch is clean — no risk of accidentally running both paths.
- The `_assert_single_actuator_writer()` prevents the IsaacLab actuator net from conflicting with the custom ROM writer.
- The FD observer-only path explicitly zeros its output before storing in the observation buffer — no effort leakage to simulation.

### Dead code concerns

- `apply_action_full_pipeline()` is defined but never called in the ROM-enabled config.
- `LearnedControlPipeline` exists as a stand-alone alternative but is unused by the IsaacLab env integration.
- `ReducedHydraulicActuatorROM` is still importable and has a cfg path but `use_final_causal_cylinder_rom=True` in all current task-space configs means it is never instantiated.

---

## Section L: Documentation Consistency

### Documentation files found

| File | Focus | Consistent? |
|------|-------|-------------|
| [`ACTIVE_PIPELINE_FILES.md`](../kevin_integration/ACTIVE_PIPELINE_FILES.md) | File inventory | ✅ Matches codebase |
| [`FULL_HYDRAULIC_PIPELINE.md`](../kevin_integration/FULL_HYDRAULIC_PIPELINE.md) | Architecture overview | ✅ Describes ROM end-to-end |
| [`ISAACLAB_CONTROL_CONTRACT.md`](../kevin_integration/ISAACLAB_CONTROL_CONTRACT.md) | Control modes/contracts | ✅ Matches cfg |
| [`TASKSPACE_RECONCILIATION_REPORT.md`](../kevin_integration/TASKSPACE_RECONCILIATION_REPORT.md) | Task-space audit | ✅ Documents 62D/73D split |
| [`RRLAB_TASKSPACE_REFERENCE_TRACE.md`](../kevin_integration/RRLAB_TASKSPACE_REFERENCE_TRACE.md) | Reference tracing | ✅ Traces vehicle conventions |
| [`RRLAB_VEHICLE_AND_REFERENCE_AUDIT.md`](../kevin_integration/RRLAB_VEHICLE_AND_REFERENCE_AUDIT.md) | Vehicle audit | ✅ Matches rrlab_reference_conventions |
| [`PPO_ACTION_REFERENCE_ALIGNMENT.md`](../kevin_integration/PPO_ACTION_REFERENCE_ALIGNMENT.md) | Action space audit | ✅ |
| [`rom_cylinders/ROM_DATA_CONTRACT.md`](../kevin_integration/rom_cylinders/ROM_DATA_CONTRACT.md) | ROM data contract | ✅ Matches rom_data_contract.py |
| [`rom_cylinders/ROM_CONTROL_CONTRACT.md`](../kevin_integration/rom_cylinders/ROM_CONTROL_CONTRACT.md) | ROM control contract | ✅ |
| [`rom_cylinders/README.md`](../kevin_integration/rom_cylinders/README.md) | ROM overview | ✅ |
| [`rom_cylinders/FIRST_DELIVERABLE_SUMMARY.md`](../kevin_integration/rom_cylinders/FIRST_DELIVERABLE_SUMMARY.md) | Deliverable summary | ✅ |
| [`results/runtime_rom_motion_contract.md`](../kevin_integration/results/runtime_rom_motion_contract.md) | ROM motion results | ✅ |
| [`results/full_pipeline_preflight_contract.md`](../kevin_integration/results/full_pipeline_preflight_contract.md) | Preflight results | ✅ |
| [`results/static_signal_chain_audit.md`](../kevin_integration/results/static_signal_chain_audit.md) | Signal chain audit | ✅ |
| [`DEBUG_CLEANUP_REPORT.md`](../kevin_integration/DEBUG_CLEANUP_REPORT.md) | Debug/cleanup log | ✅ |

### Gaps found

1. **No documentation of the `run_learned_pipeline_preflight()` observation-dimension bug** — the only documentation is in the source code comment at line 2793 ("Temporarily switch to scripted demo mode for the warm-start."), but the side-effect on observation dimension is not noted.

2. **No explicit documentation of the FD observer-only contract** — while `_run_fd_observer_only()` is clearly named, there is no document explaining that in ROM mode the FD torque is not applied to simulation.

3. **No end-to-end data-flow diagram** — the architecture is described textually, but a diagram showing the observation → PPO → AAM → ID → ROM → position target flow would help new developers.

---

## Summary of Defects

| # | Severity | File | Line(s) | Description |
|---|----------|------|---------|-------------|
| 1 | **HIGH** | [`mulag_adaptive_rl_env.py`](../kevin_integration/tasks/mulag_adaptive_rl_env.py) | 2793-2795 | `run_learned_pipeline_preflight()` mutates `cfg.control_mode` to `"scripted_reference_demo"` then calls `reset()`, causing `_get_observations()` to select 62D when 73D is expected → crash |
| 2 | MEDIUM | [`mulag_adaptive_rl_env.py`](../kevin_integration/tasks/mulag_adaptive_rl_env.py) | 1562-1595 | `_get_observations()` switch uses `_task_control_mode()` which returns unrecognised modes literally — any unexpected string falls through to 62D default |
| 3 | LOW | [`tasks/__init__.py`](../kevin_integration/tasks/__init__.py) | — | Three task IDs alias to identical config — not a bug but confusing for users |
| 4 | LOW | [`controllers/pipeline.py`](../kevin_integration/controllers/pipeline.py) | — | `LearnedControlPipeline` is dead code (not wired into env); numpy-based pipeline is superseded by `AdaptiveRLLearnedController` |
| 5 | INFO | [`controllers/reduced_hydraulic_actuator.py`](../kevin_integration/controllers/reduced_hydraulic_actuator.py) | — | `ReducedHydraulicActuatorROM` is dead code when `use_final_causal_cylinder_rom=True` (all current configs) |

---

## Appendix: File Inventory

### Files inspected (complete read)

| File | Lines | Purpose |
|------|-------|---------|
| `tasks/mulag_adaptive_rl_env.py` | 2949 | Main environment class |
| `tasks/mulag_adaptive_rl_env_cfg.py` | 353 | Environment configurations |
| `tasks/__init__.py` | 44 | Gym task registrations |
| `tasks/task_space_reference.py` | 66 | Observation dimension constants |
| `tasks/rrlab_reference_conventions.py` | 69 | Joint/vehicle conventions |
| `rl/observation_builder.py` | 157 | 62D and 73D observation builders |
| `rl/adaptive_rl_controller.py` | 706 | Learned controller (AAM/ID/FD orchestration) |
| `rl/reward_terms.py` | 269 | Reward functions |
| `rl/action_adapter.py` | 27 | Action dimension helpers |
| `rl/torque_adapter.py` | 187 | FD torque adapter |
| `controllers/learned_models.py` | 717 | AAM/ID/FD/SD PyTorch models |
| `controllers/pipeline.py` | 138 | Legacy pipeline (unused) |
| `controllers/reduced_hydraulic_actuator.py` | 223 | Legacy ROM (unused in ROM mode) |
| `rom_cylinders/valve_conditioned_causal_rom.py` | 188 | Valve-conditioned causal ROM model |
| `rom_cylinders/multirate_rom_wrapper.py` | 196 | 0.01s/0.05s multirate wrapper |
| `rom_cylinders/cylinder_coordinate_adapter.py` | 399 | Simscape↔Isaac coordinate mapping |
| `rom_cylinders/rom_data_contract.py` | 130 | ROM data contract constants |
| `rom_cylinders/exported_valve_conditioned_causal_rom/best_valve_conditioned_causal_rom_metadata.json` | 1092 | Full ROM metadata |
| `scripts/train_mulag_adaptive_rl.py` | 478 | Training entry point |
| `scripts/play_mulag_adaptive_rl.py` | 340 | Playback/eval entry point |

### Reference files inspected

| File | Lines | Purpose |
|------|-------|---------|
| `extensions/rrlab_tasks/.../obstacle_avoidance_env_cfg.py` | 577 | RRLAB obstacle avoidance env config |
| `extensions/rrlab_tasks/.../config/mulag/joint_pos_env_cfg.py` | 200 | Mulag-specific env config |
| `extensions/rrlab_tasks/.../mdp/commands.py` | 216 | Pose command generator |
| `extensions/rrlab_tasks/.../mdp/commands_cfg.py` | 67 | Command cfg |

---

*End of report. All findings are based on static source code analysis without execution.*
