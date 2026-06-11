# Corrected Addendum — Kevin Integration Codebase Inspection

**Date:** 2026-06-11  
**Scope:** Corrects errors in `plans/integration_codebase_inspection_report.md` for the 9 points specified.  
**Method:** Static source-code analysis of `kevin_integration/`.

---

## 1. Preflight Fix — Acceptable Minimal Correction

The original report correctly identified the crash at [`run_learned_pipeline_preflight()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:2778) lines 2793–2795: mutating `cfg.control_mode = "scripted_reference_demo"` causes the subsequent `self.reset()` to select the 62D observation path via [`_get_observations()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1562), which then fails because `observation_space` was locked to 73 at construction.

### Acceptable minimal fix

The only acceptable change is:

1. **Keep** `cfg.control_mode = "rl_task_space_tracking"` — do not mutate it.
2. **Keep** the canonical observation contract at 73D — do not bypass the dimension assertion at line 1591–1594.
3. **Do not** bypass the dimension assertion or temporarily allow a 62D observation.
4. **Add an internal preflight-only scripted-action flag** (e.g., `self._preflight_use_scripted_action: bool`).
5. **Use that flag** to replace only the source of `Action[4]` — specifically, inside [`_prepare_high_level_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:876) or the action computation branch for `rl_task_space_tracking`, check the flag and, if set, substitute `self._scripted_reference_action(...)` for the PPO/RL action tensor.
6. **Restore the flag** with a `try/finally` block in [`run_learned_pipeline_preflight()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:2809–2824) — the `finally` at line 2824 already restores `cfg.control_mode`; the flag restoration should co-locate there.

### Exact functions requiring editing

| Function | File | Line | Edit |
|----------|------|------|------|
| [`run_learned_pipeline_preflight()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:2778) | `mulag_adaptive_rl_env.py` | 2793–2795, 2809–2824 | Replace `cfg.control_mode` mutation with flag set; add `try/finally` for flag restore |
| [`_prepare_high_level_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:876) | `mulag_adaptive_rl_env.py` | ~890–903 | Check `self._preflight_use_scripted_action` and call `_scripted_reference_action()` instead of using RL action |
| **Init block** | `mulag_adaptive_rl_env.py` | ~158 | Declare `self._preflight_use_scripted_action: bool = False` |

No changes to [`_get_observations()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1562), [`_task_control_mode()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:397), the 73D assertion, [`build_task_space_rl_obs()`](kevin_integration/rl/observation_builder.py:73), or [`build_rl_obs()`](kevin_integration/rl/observation_builder.py:35).

---

## 2. Actual Cylinder Writer — `_apply_action()` Details

### Exact normal final-ROM call

At [`_apply_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1525) line 1532–1534:

```python
self.robot.set_joint_position_target(
    self._rom_cylinder_pos_target,          # target tensor: shape (num_envs, 4)
    joint_ids=self._command_joint_ids       # joint ID tensor: 4 IDs
)
```

- **Target tensor:** `self._rom_cylinder_pos_target` — shape `(num_envs, 4)`, dtype float32.
- **Joint ID tensor:** `self._command_joint_ids` — resolved at init from [`OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES`](kevin_integration/tasks/rrlab_reference_conventions.py:14) = `("Drehzapfen_joint", "Ausleger_I_joint", "Ausleger_II_joint", "Messerkopf_Schwenk_joint")`.
- **Number of joints:** 4 (four prismatic cylinders only). **Does not include `Messerkopf_joint`.**

This corrects the original report's claim that the writer "writes position targets to the 4 ROM cylinder prismatic joints + Messerkopf_joint (joint 5, which follows via coupling)." The fifth joint (`Messerkopf_joint`) is **not** directly written — it moves via the mechanical coupling implemented at [`_apply_joint5_coupling()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:104) or through USD physics constraints. The original report also incorrectly claimed a loop over five joints.

### Complete inventory of `set_joint_position_target` and `set_joint_effort_target` calls

#### `set_joint_position_target` calls

| # | Location | Code | Target | Joint IDs | Classification |
|---|----------|------|--------|-----------|----------------|
| 1 | [`_apply_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1527) | `self._preflight_direct_target` | Direct preflight pulse | `self._command_joint_ids` (4) | **PREFLIGHT** |
| 2 | [`_apply_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1533) | `self._rom_cylinder_pos_target` | ROM position target | `self._command_joint_ids` (4) | **CANONICAL RUNTIME** (use_final_causal_cylinder_rom) |
| 3 | [`_apply_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1537) | `self._rom_cylinder_pos_target` | Reduced Hydraulic ROM target | `self._command_joint_ids` (4) | **REACHABLE_BUT_DISABLED** (use_reduced_hydraulic_rom) |
| 4 | [`_apply_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1553) | `q_now + position_scale * delta_cmd` | Position-delta target | `self._command_joint_ids` (4) | **REACHABLE_BUT_DISABLED** (position_delta + !use_fd_effort) |

#### `set_joint_effort_target` calls

| # | Location | Code | Target | Joint IDs | Classification |
|---|----------|------|--------|-----------|----------------|
| 5 | [`_apply_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1557) | `self._current_torque` | FD effort | `self._command_joint_ids` (4) | **REACHABLE_BUT_DISABLED** (effort or use_fd_effort) |

**Note:** Call #4 (position-delta) and #5 (effort) are reachable only if `use_final_causal_cylinder_rom=False` AND `use_reduced_hydraulic_rom=False` (i.e., the early-return guards at lines 1532–1538 are bypassed). In the canonical `MulagHydraulicROMTaskSpaceEnvCfg`, `use_final_causal_cylinder_rom=True`, so calls #3, #4, and #5 are all disabled at runtime by the early return at line 1534.

No `set_joint_effort_target` or `set_joint_position_target` calls exist outside [`_apply_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1525) in the main env loop. Vehicle wheels use [`write_joint_velocity_to_sim()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:958) — a separate API.

---

## 3. FD Observer Semantics

### `fd_tau_hat` remains nonzero and is stored for observation, reward, and telemetry

In [`_run_fd_observer_only()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1034–1062):

```python
# Line 1044: FD model runs and produces torque estimate
fd_tau = self._controller.run_fd_observer(q, qdot, dP, valve_cmd)

# Lines 1049-1051: fd_tau_hat IS stored (nonzero)
self._current_fd_simscape_output = fd_tau.detach().clone()   # line 1049
self._current_tau_fd_adapted = fd_tau.detach().clone()       # line 1050
self._fd_tau_hat = fd_tau.detach().clone()                   # line 1051
```

Then `_fd_tau_hat` is consumed in:

- **Observation:** [`_get_observations()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1582) — `self._fd_tau_hat` is passed to `build_task_space_rl_obs()` and appears in the 73D observation at slice `[68:72]`.
- **Reward:** [`_get_rewards()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1610–1611) — `self._fd_tau_hat_normalized` (derived from `fd_tau_hat` via normalization) is used in the FD load penalty.
- **Telemetry:** [`_get_debug_info()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1891) — `debug["fd_tau_hat"] = torch.mean(self._fd_tau_hat.detach(), dim=0)`.

### Physically applied torque buffers are zeroed

Immediately after storing `fd_tau_hat` (lines 1056–1062):

```python
self._current_torque.zero_()              # line 1056 — the actual applied torque
self._current_tau_applied_raw.zero_()     # line 1057
self._current_tau_isaac_filtered.zero_()  # line 1058
self._current_tau_isaac_clamped.zero_()   # line 1059
self._torque_clamp_fraction.zero_()       # line 1060
self._torque_adapter_clamp_fraction.zero_()  # line 1061
self._torque_rate_limited_fraction.zero_()   # line 1062
```

Since `_apply_action()` dispatches to `self.robot.set_joint_position_target(...)` at line 1533 when `use_final_causal_cylinder_rom=True`, and never reaches the `set_joint_effort_target` path at line 1557, the zeroed `_current_torque` is never physically applied.

### FD effort actuation is disabled in canonical ROM mode

The `MulagHydraulicROMTaskSpaceEnvCfg` at [`mulag_adaptive_rl_env_cfg.py:317`](kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py:317) sets:

```python
use_fd_effort = False          # inherited from base class line 50
use_fd_effort_actuation = False  # inherited from base class line 56
```

The assertion at [`_assert_single_actuator_writer()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:353–364) raises `RuntimeError` if both ROM position-target and FD effort actuation are enabled simultaneously.

---

## 4. ROM Feature Assembly — 26D vs 17-Term Distinction

The original report conflated the 26D runtime input vector with the internal 17-term-per-cylinder regression basis. The correct distinction:

### 26D raw runtime ROM input — assembled by `_final_causal_rom_feature_row()`

At [`_final_causal_rom_feature_row()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1376) lines 1400–1409:

```python
features = torch.cat([
    valve_cmd,            # 4  (ValveCmd_1..4)
    rom_pressure_input,   # 4  (dP_1..4)
    extension,            # 4  (Extension_1..4) — mapped cylinder state
    extension_velocity,   # 4  (ExtensionVelocity_1..4)
    q,                    # 5  (JointPos_1..5) — mechanical joint state
    qdot,                 # 5  (JointVel_1..5)
], dim=-1)  # Total: 26
```

This 26D vector is defined by [`CORE_INPUT_FEATURE_NAMES`](kevin_integration/rom_cylinders/rom_data_contract.py:6) and validated at line 1411–1415.

The environment does **not** directly assemble the 17 regression terms. It only assembles this 26D feature vector and passes it to the ROM wrapper.

### 17-term structured regression basis — computed inside the exported model

Inside [`ValveConditionedCausalCylinderROM._features()`](kevin_integration/rom_cylinders/valve_conditioned_causal_rom.py:69), the 26D input is converted to 17 terms **per cylinder**:

```python
for i in range(4):
    rows.append(torch.stack([
        ones,                 # 1  — bias
        u[:, i],              # 1  — valve linear (u)
        u[:, i] * |u[:, i]|,  # 1  — valve quadratic (u*|u|)
        u[:, i] * dp[:, i],   # 1  — valve-pressure interaction (u*dP)
        dp[:, i],             # 1  — pressure (dP)
        vel[:, i],            # 1  — extension velocity (v)
        ext[:, i],            # 1  — extension (x)
        *q.T,                 # 5  — JointPos_1..5 (centered)
        *qv.T,                # 5  — JointVel_1..5
    ], dim=-1))  # Total per cylinder: 1+1+1+1+1+1+1+5+5 = 17
```

The 4 × 17 = 68 regression coefficients (stored as `coefficients` buffer shape `(4, 17)`) multiply these terms. The summed output per cylinder is bounded by a tanh correction envelope, then added to the kinematic baseline (`extension_velocity * dt`).

**Key correction:** The environment never directly computes the 17-term basis. It only provides the 26D raw input. The 17-term expansion is internal to the exported TorchScript/`nn.Module` model.

---

## 5. Mechanical Versus Cylinder State Contract

### Names and IDs

| Property | Value | Source |
|----------|-------|--------|
| `state_joint_names` | `("Drehzapfen_joint", "Ausleger_I_joint", "Ausleger_II_joint", "Messerkopf_Schwenk_joint", "Messerkopf_joint")` — 5 joints | [`rrlab_reference_conventions.py:6`](kevin_integration/tasks/rrlab_reference_conventions.py:6) |
| `command_joint_names` | Same as `state_joint_names[:4]` — 4 joints | [`rrlab_reference_conventions.py:14`](kevin_integration/tasks/rrlab_reference_conventions.py:14) |
| `_state_joint_ids` | Resolved from 5 `state_joint_names` at init | [`mulag_adaptive_rl_env.py:72`](kevin_integration/tasks/mulag_adaptive_rl_env.py:72) |
| `_command_joint_ids` | Resolved from 4 `command_joint_names` at init | [`mulag_adaptive_rl_env.py:73`](kevin_integration/tasks/mulag_adaptive_rl_env.py:73) |

### Assertion

At [`_pre_physics_step_final_causal_rom()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1192):

```python
if tuple(self.cfg.command_joint_names) != tuple(self.cfg.state_joint_names[:4]):
    raise RuntimeError("Final causal ROM requires the first four state joints to be the four hydraulic cylinders.")
```

This **enforces** that `command_joint_names == state_joint_names[:4]`.

### How `q[5]` and `qdot[5]` enter AAM, ID, FD, and ROM

- [`_read_joint_state()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:593) reads from `_state_joint_ids` → 5 mechanical arm joints (raw Isaac USD joint coordinates). These `q[5]`, `qdot[5]` are passed to:
  - [`compute_z_arm()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1567) → AAM
  - [`_pre_physics_step_final_causal_rom()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1183) → ID (via `apply_action_valve_only`), FD (via `run_fd_observer`), and ROM feature row (as `JointPos[5]`, `JointVel[5]`)
  - [`_get_observations()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1563) → 73D observation `mechanical_q[5]`, `mechanical_qdot[5]`

### How `Extension[4]` separately enters the ROM

- [`_read_cylinder_state()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:598) reads from `_command_joint_ids` → 4 cylinder joints (same physical joints as `state_joint_names[:4]`).
- These 4 raw Isaac coordinates are mapped through [`CylinderCoordinateMapping`](kevin_integration/rom_cylinders/cylinder_coordinate_adapter.py) at [`_mapped_cylinder_state()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1354) to produce `extension[4]` and `extension_velocity[4]` in Simscape coordinates.
- These mapped values enter the ROM feature row as `Extension_1..4` and `ExtensionVelocity_1..4`.

### Answer to the exact question

> Are the runtime `JointPos[5]` and `JointVel[5]` genuinely mechanical arm coordinates, or are their first four channels currently the same prismatic cylinder coordinates used for `Extension[4]`?

**CONFIRMED MATCH — by design, with distinct coordinate representations.**

- `JointPos[5]` and `JointVel[5]` are **genuine mechanical arm coordinates**: raw Isaac USD joint readings from the 5 arm joints.
- `Extension[4]` and `ExtensionVelocity[4]` are **cylinder-specific mapped coordinates**: the same 4 physical DOFs as the first 4 channels of `JointPos[5]`, but transformed through `CylinderCoordinateMapping` into Simscape extension space (meters of cylinder stroke, with sign/offset mapping).
- The assertion at line 1192 explicitly enforces `command_joint_names == state_joint_names[:4]`, guaranteeing the first 4 state joints map to the same USD degrees of freedom as the 4 command joints.
- This duality is **by design**: the ROM model was trained on Simscape data where both raw mechanical joint position (JointPos) and mapped cylinder extension (Extension) are distinct and necessary signals.

---

## 6. Vehicle Wheel Signs

### Does the implementation send the same signed target to all four wheels?

**YES.** In [`_apply_rrlab_vehicle_command()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:930), lines 952–960:

```python
omega = (sampled_speed_kmh / 3.6) / float(self.cfg.vehicle_wheel_radius_m)
self._vehicle_wheel_velocity_target[env_ids] = omega.reshape(-1, 1).expand(
    -1, len(self._vehicle_wheel_joint_ids)
)
self.robot.write_joint_velocity_to_sim(
    self._vehicle_wheel_velocity_target,
    joint_ids=self._vehicle_wheel_joint_ids,
)
```

A single scalar `omega` (computed from forward speed) is expanded to a `(num_envs, 4)` tensor — all four wheel joints receive the **identical signed velocity target**. No sign reversal is applied for left vs right wheels.

### Does this match the RRLAB source numerically?

The `kmh_to_wheel_rad_per_sec()` conversion from [`rrlab_reference_conventions.py`](kevin_integration/tasks/rrlab_reference_conventions.py:48) is source-equivalent to the RRLAB obstacle task. Both use `(speed_kmh / 3.6) / wheel_radius_m` with `wheel_radius_m = 0.575`. The numerical value matches.

However, **the RRLAB obstacle task's USD asset** may define wheel joint axes with opposing signs for left vs right wheels (e.g., a positive velocity command producing forward rotation on the right side but backward rotation on the left). Without inspecting the USD articulation axis directions, **it is not confirmed** that both codebases produce the same vehicle motion direction with the same sign convention.

### Have the USD axis directions been statically verified?

**NO.** The USD file [`unimog_mulag_fme500_cylinder_as_actuator.usd`](extensions/rrlab_assets/data/Robots/unimog_mulag_fme500_cylinder_as_actuator.usd) has not been inspected for wheel joint axis directions. The [`MULAG_CFG`](extensions/rrlab_assets/rrlab_assets/mulag.py:12) defines an `ImplicitActuatorCfg` with the four wheel joint names but leaves stiffness/damping as `None` (deferring to USD). No static verification of axis signs has been performed.

### What runtime measurement is still required?

A runtime visual or logged confirmation that a positive velocity command produces forward vehicle motion in the same sense as the RRLAB obstacle task. Specifically:
1. Command `omega > 0` to all four wheels.
2. Observe vehicle direction of travel in Isaac Sim.
3. Verify no wheel spins backwards relative to forward direction (which would indicate missing sign flip for left/right wheel pairs in the USD axis convention).

---

## 7. Legacy-Path Reclassification

The original report used "dead code" incorrectly for paths that are merely disabled by configuration. Corrected classifications:

| Path | Component | File | Classification | Rationale |
|------|-----------|------|---------------|-----------|
| **Final causal cylinder ROM** | `ValveConditionedCylinderROM` via `_pre_physics_step_final_causal_rom()` | `mulag_adaptive_rl_env.py:1181` | **ACTIVE** | Canonical path in `MulagHydraulicROMTaskSpaceEnvCfg` |
| **Placeholder `ReducedHydraulicActuatorROM`** | `_pre_physics_step_reduced_hydraulic_rom()` | `mulag_adaptive_rl_env.py:1420` | **REACHABLE_BUT_DISABLED** | Guarded by `use_reduced_hydraulic_rom=True`; constructor always instantiates it at line 144. Disabled in canonical config. |
| **Direct ValveCmd position-delta path** | `position_delta_source == "id_valve"` | `mulag_adaptive_rl_env.py:1544–1553` | **REACHABLE_BUT_DISABLED** | Guarded by early-return at line 1534 when `use_final_causal_cylinder_rom=True`. Reachable if both ROM flags are `False`. |
| **Direct-action position-delta path** | `position_delta_source == "direct_action"` | `mulag_adaptive_rl_env.py:1542–1553` | **REACHABLE_BUT_DISABLED** | Same guard as above. Also requires `cfg.position_delta_source = "direct_action"` which is not the canonical setting. |
| **FD effort path** | `set_joint_effort_target` via `_pre_physics_step` effort branch | `mulag_adaptive_rl_env.py:1125–1149, 1556–1558` | **REACHABLE_BUT_DISABLED** | Guarded by `low_level_mode == "effort" or use_fd_effort`. Both `False` in canonical config. |
| **Old 62D task** | `MulagAdaptiveRLJointReachEnvCfg` | `mulag_adaptive_rl_env_cfg.py:30` | **REACHABLE_BUT_DISABLED** | Base config with `control_mode="position_delta"` and `observation_space=62`. Used by `Kevin-Mulag-AdaptiveRL-v0` alias, but task-space config overrides to 73D. The 62D observation path is reachable only if a config with `control_mode != "rl_task_space_tracking"` is used. |
| **`LearnedControlPipeline`** (standalone numpy pipeline) | `pipeline.py` | `kevin_integration/controllers/pipeline.py:35` | **LEGACY** | NumPy-based predecessor to `AdaptiveRLLearnedController`. Not wired into the IsaacLab env. Still importable, not dead — could be used for offline validation. |
| **`apply_action_full_pipeline()`** (AAM→ID→FD→torque) | `adaptive_rl_controller.py:458` | `kevin_integration/rl/adaptive_rl_controller.py` | **REACHABLE_BUT_DISABLED** | Called only from the effort/low-level path in `_pre_physics_step` (line 1126), which is guarded by the ROM early-returns. Method exists and works, but is not invoked in canonical ROM mode. |

---

## 8. Corrected Risk Ranking

| Rank | Issue | Evidence |
|------|-------|----------|
| **BLOCKING** | 62D/73D preflight crash — `run_learned_pipeline_preflight()` mutates `cfg.control_mode` causing dimension mismatch crash before PPO can start | [`mulag_adaptive_rl_env.py:2793–2795`](kevin_integration/tasks/mulag_adaptive_rl_env.py:2793) |
| **HIGH** | Mechanical-vs-cylinder feature meaning — the duality of `JointPos[1..4]` vs `Extension[1..4]` (same physical DOFs, different coordinate spaces) must match training-time feature semantics. Assertion at line 1192 confirms structural match, but the `CylinderCoordinateMapping` sign/offset values have not been validated against training data. See point 5. | [`mulag_adaptive_rl_env.py:1192`](kevin_integration/tasks/mulag_adaptive_rl_env.py:1192), [`cylinder_coordinate_adapter.py`](kevin_integration/rom_cylinders/cylinder_coordinate_adapter.py) |
| **MEDIUM** | Vehicle wheel signs unverified — same signed `omega` sent to all four wheels; USD axis directions not statically checked; runtime measurement required | [`mulag_adaptive_rl_env.py:952–960`](kevin_integration/tasks/mulag_adaptive_rl_env.py:952) |
| **LOW** | Three task IDs alias to identical config — not a runtime bug but confusing | [`tasks/__init__.py`](kevin_integration/tasks/__init__.py) |
| **LOW** | `LearnedControlPipeline` (numpy pipeline) is legacy, not wired into env | [`controllers/pipeline.py`](kevin_integration/controllers/pipeline.py) |

---

## 9. One Narrow Edit Task

**Task:** Keep the canonical 73D observation contract while using a scripted `Action[4]` source during learned-pipeline preflight.

**Scope:** Only [`mulag_adaptive_rl_env.py`](kevin_integration/tasks/mulag_adaptive_rl_env.py).

**Changes:**
1. Add `self._preflight_use_scripted_action: bool = False` at init (~line 158).
2. In [`run_learned_pipeline_preflight()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:2778): replace `self.cfg.control_mode = "scripted_reference_demo"` with `self._preflight_use_scripted_action = True`. Restore in `finally`.
3. In [`_prepare_high_level_action()`](kevin_integration/tasks/mulag_adaptive_rl_env.py:876): after the RL action branch, check `self._preflight_use_scripted_action` and, if set, override `self._current_action` with the output of `_scripted_reference_action()`.

**Excluded (not part of this task):**
- No changes to observations, rewards, or 73D contract.
- No changes to `_task_control_mode()`, `_get_observations()`, or the dimension assertion.
- No changes to vehicle control, ROM model, FD, SD, or coordinate mapping.
- No config changes, model retraining, or refactoring beyond the three lines described.

---

UNDERSTOOD — ready for a narrowly scoped correction task
