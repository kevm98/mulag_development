# Current PPO1 Pipeline Summary

**Auto-generated from codebase inspection on 2026-07-08/09.**
See `kevin_integration/new_observations_pipeline/ppo1/README.md` for the authoritative description.

## Architecture

```
PPO action (ΔEE, base-frame Y/Z, [-1,1])
  → × ee_action_scale_m
  → empirical J_eff (damped least-squares)
  → desired cylinder-space motion dx_des
  → ID model (VelocityToValveMLP: [dx, v, x, xd] → valve)
  → FD model (sequence GRU: [valve, x, xd] → Δextension)
  → closed-loop horizon rollout (H steps)
  → cylinder position targets (smooth-ramped)
  → PhysX implicit position drives
```

## What the PPO Action Means

- **Dimension**: 4 (cylinder-space mode) or 3 (task-space mode: base-frame Y, Z)
- **Scale**: `ee_action_scale_m = 0.05` (task-space) or `action_scale = 0.02` (cylinder-space)
- **Range**: `[-1, 1]` → multiplied by scale to get cylinder delta (m) or EE delta (m)
- **Interpretation**: PPO action IS a desired delta-EE command in base frame (task-space mode).
  It is NOT a joint command and NOT a direct cylinder command.

## Action Flow (Task-Space Mode)

1. PPO outputs action ∈ [-1,1]^3 (base-Y, base-Z for ground tracking)
2. × ee_action_scale_m → desired EE delta in base frame
3. HighLevelEEController.map_ee_to_cylinder():
   - Uses measured J_eff [3×4] via damped least-squares
   - Clips per-cylinder (0.0015 m default) and global (0.003 m default)
   - Temporal smoothing (EMA α=0.7)
4. Result: dx_des_H [B, 4] — desired cylinder delta in Simscape extension space
5. ID model: [dx_des, v_des, x_current, xd_current] → valve command
6. FD model: [valve, x, xd] → predicted delta extension
7. Horizon rollout: H-step closed loop → accumulated fd_delta
8. fd_gains × fd_delta → delta_cmd_H (clipped to ±0.02 m)
9. Smooth ramp over H env steps onto cylinder position target

## Where Actions Are Clipped

- `ee_per_cyl_clip_m = 0.0015` — per-cylinder delta clip in EE mapper
- `ee_global_clip_m = 0.003` — global norm clip in EE mapper
- `DELTA_CLIP_M = 0.02` — horizon command clip in `horizon_idfd_controller.py`
- `action_scale = 0.02` — PPO action scale (cylinder space)
- Joint limit margin: `horizon_cylinder_limit_margin = 0.002`

## Where Observations Are Built

- `ppo1/observations_adapter.py`: `build_horizon_observation()` and `build_high_level_observation()`
- Observation includes:
  - Joint position history (5 joints × 3 history steps)
  - Joint velocity history
  - Distance-to-goal history
  - Joint torques
  - Last action
  - Vehicle pose and velocity

## Joint Position Order

From `rrlab_reference_conventions.py`:
```
[0] Drehzapfen_joint         — cylinder 0 (base rotation)
[1] Ausleger_I_joint          — cylinder 1 (boom 1)
[2] Ausleger_II_joint         — cylinder 2 (boom 2)
[3] Messerkopf_Schwenk_joint  — cylinder 3 (cutter swivel)
[4] Messerkopf_joint          — passive/coupled (no cylinder)
```

## Cylinder Variable Order

From `cylinder_coordinate_adapter.py` and `cylinder_coordinate_mapping.json`:
```
[0] Drehzapfen_joint
[1] Ausleger_I_joint
[2] Ausleger_II_joint
[3] Messerkopf_Schwenk_joint
```

## Cylinder Coordinate Mapping

All 4 cylinders: `sign = -1` (verified, confidence: HIGH).

Formula (nominal_sign_offset mode):
```
x_simscape = x_nominal + sign * (q_isaac - q_nominal)
           = x_nominal - (q_isaac - q_nominal)

delta_x_simscape = sign * delta_q_isaac = -delta_q_isaac
```

This means: **positive Isaac prismatic joint motion corresponds to negative Simscape cylinder extension change.** When the PhysX cylinder extends, the Simscape model sees it as retraction.

Nominal values:
| Cylinder | x_nominal (Simscape) | q_nominal (Isaac) | Sign |
|---|---|---|---|
| Drehzapfen | 0.3352 | -0.0029 | -1 |
| Ausleger_I | 0.5414 | 0.0458 | -1 |
| Ausleger_II | 0.4934 | 0.0245 | -1 |
| Messerkopf_Schwenk | 0.5867 | 0.0423 | -1 |

## Where ID Input Is Built

`joint_to_cylinder_adapter.py::JointToCylinderAdapter.build_id_input()`:
```
id_input = [x_current, xdot_current, dP, force, delta_x_des, delta_xdot_des]  (24-D)
```
All in Simscape cylinder extension coordinates after conversion.

## Where FD/ROM Input Is Built

- FD model: `sequence_fd.push_current_state(valve, extension, extension_velocity)`
- Uses GRU history of [valve, extension, extension_velocity] → predicts delta_extension
- Model trained on Simscape hydraulic data
- Runtime: `closed_loop_idfd_rollout()` in `idfd_rollout.py`

## Where AAM Latent Is Built

Not applicable in current pipeline. The FD model uses explicit GRU history, not a learned latent.

## Where IsaacLab Command Is Finally Applied

`mulag_upper_level_env.py::_apply_action()`:
```python
self.robot.set_joint_position_target(
    self._cylinder_pos_target,
    joint_ids=self._cylinder_joint_ids,
)
```
This sets **position targets** on the 4 cylinder prismatic joints.

## Command Type

- **Position targets** (implicit position drives)
- NOT velocity or effort — the PhysX solver closes the position loop
- Drive stiffness models hydraulic lock (closed valve = rigid oil column)

## Implicit PhysX PD

Yes, implicit position drives use PhysX's internal PD controller:
- Stiffness: K = [7.5e7, 3e7, 5e6, 5e6] (from drive stiffness bracket)
- Damping: tuned for critical-ish damping
- This is NOT the explicit force PID — the explicit PID was abandoned as unstable

## Key Flags

- `lowlevel_actuator_backend = "implicit_position"` — production default
- `horizon_reference_space_target` — integration fix (change 16): accumulates commands onto persistent reference instead of re-anchoring to measured (sagged) position
- `horizon_pin_root_pose = False` — root pinning removed (was pumping energy)

## Known Issues

1. **Reference-space integration**: Without `--reference_space_target`, the measured-space integration causes a gravity ratchet (tool drifts 0.38 m with zero command). FIXED by change 16.
2. **Sign convention**: All 4 cylinders have sign=-1. Positive Isaac q → negative Simscape x. This is correctly handled by the coordinate adapter.
3. **Joint 5 coupling**: `Messerkopf_joint` is configured with `enable_joint6_coupling=True, coupling_a=1.0`. It may be mechanically coupled to joint 4.
4. **Stage0 local reach**: Broken, not validated. Do not use.
5. **Timescale factor 3**: ID/FD models hydraulic motion at 0.05 s but ramp executes at 0.0167 s. Absorbed by J_eff feedback but may cause scaling errors.

## Critical File Paths

| File | Role |
|---|---|
| `ppo1/horizon_idfd_env_wrapper.py` | Horizon wrapper: J_eff, ID/FD rollout, rewards |
| `ppo1/mulag_upper_level_env.py` | Base env: robot, actuation, cylinder targets |
| `ppo1/mulag_upper_level_env_cfg.py` | All config: actuators, guards, rewards |
| `ppo1/joint_to_cylinder_adapter.py` | Isaac ↔ Simscape coordinate conversion |
| `low_level_controller/high_level_ee_controller.py` | J_eff measurement + damped-LS |
| `low_level_controller/horizon_idfd_controller.py` | Horizon command computation |
| `low_level_controller/idfd_rollout.py` | Closed-loop ID/FD rollout |
| `rom_cylinders/cylinder_coordinate_adapter.py` | Cylinder coordinate mapping class |
| `rom_cylinders/cylinder_coordinate_mapping.json` | Mapping parameters (signs, nominals) |
| `tasks/rrlab_reference_conventions.py` | Joint name definitions |
