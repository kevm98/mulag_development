# Current PPO Trajectory Audit

**Date:** 2026-07-13
**Git commit:** fd1f674156094246ddc969fc86a73ccef64c9349
**Branch:** codex/ppo1-stage1-debug-residual
**IsaacLab:** v2.3.2 (commit f4aa17f87e2)

## 1. How the current PPO path is generated

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py:44-49`

```python
planner = GroundPathPlanner(str(getattr(cfg, "residual_map", _DEFAULT_MAP)))
path = planner.arc()
```

`GroundPathPlanner.arc()` in [ground_paths.py](/home/rrlab/rrlab_mulag/ID_FD_pipline/dls_solver/ground_paths.py):
1. Loads `ground_band_map.json` (441 physics-settled poses at cylinder grid points)
2. Builds a spatial distance matrix with edge threshold 0.75 m and cylinder-edge threshold 0.12 m
3. Finds the largest connected component (440 of 441 nodes)
4. Computes the graph diameter — the longest shortest path between any two nodes
5. **Result: 7 hops, 3.801 m cumulative arc length**
6. Resamples at 2 cm spacing → 192 waypoints with `GroundPath(goals, cylinder_seeds, direct_targets, ...)`

**NUMERICAL EVIDENCE (computed from ground_band_map.json):**

| Property | Value |
|---|---|
| Source samples | 441 |
| Connected component | 440 nodes |
| Diameter path hops | 7 |
| Raw path length | 3.801 m |
| Resampled waypoints | 192 |
| Path step spacing | 0.02 m |
| Unique X span | [3.524, 6.494] = 2.970 m |
| Unique Y span | [-3.398, -1.723] = 1.675 m |
| Unique Z span | [1.311, 1.422] = 0.111 m |
| Cutter world Z range | [-0.099, 0.099] |

The 7 diameter hops have inter-node distances: 0.649, 0.695, 0.728, 0.539, 0.737, 0.454 m. These are linearly interpolated to produce the 192-waypoint dense path.

## 2. Continuous interpolation vs discrete jumps

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py:63-73` (`_update_residual_trajectory`)

```python
progress_steps = torch.clamp(self.episode_length_buf - settle, min=0)
distance = progress_steps.float() * float(self.step_dt) * speed
idx = torch.clamp(
    (distance / max(self._path_step_m, 1.0e-9)).long(),
    0,
    self._path_goals.shape[0] - 1,
)
self._ee_goal_local = self._path_goals[idx]
```

The goal is selected by **flooring the continuous distance to an integer waypoint index**. At 2 cm spacing and 0.045 m/s, the goal advances by one waypoint every 0.44 s. There is **no interpolation between waypoints** — the goal jumps discretely. A 2 cm jump per 0.44 s is small enough to be tracked, but it is not continuous.

The path itself is spatially continuous (linear interpolation between the 7 diameter nodes), but temporally the EE goal is discrete-stepped.

## 3. One-way vs periodic

The trajectory is **strictly one-way, forward only**. The `_update_residual_trajectory` function computes:
```python
idx = torch.clamp((distance / path_step).long(), 0, n_waypoints - 1)
```

There is no modulo, no ping-pong, no triangle wave. When `distance > path_length`, the index saturates at `n_waypoints - 1`. The goal freezes. There is no return stroke.

**RUNTIME EVIDENCE:** At 0.045 m/s for 28.3 s of motion (after 6.7 s settle), the goal travels 1.27 m. This is only 34% of the 3.8 m path — the goal never reaches the end, so saturation does not occur. But the goal is still a one-way advance; the arm simply tracks a forward-crawling target.

## 4. Where path progress is updated

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py:112`, called from `_pre_physics_step`

`_update_residual_trajectory()` is called **every physics step** for all environments. Progress is purely time-based:

```python
progress_steps = self.episode_length_buf - settle_steps  # offset by settle duration
distance = progress_steps * step_dt * 0.045  # meters along path
```

There is **no feedback from actual tool tracking** — the goal advances at a fixed rate regardless of whether the arm follows.

## 5. How settling is implemented

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py:114-116, 129-141`

```python
settle = int(getattr(self.cfg, "residual_settle_steps", 150))  # = 400
settling = self.episode_length_buf < settle
# ...
self._residual_action[settling] = 0.0  # zero residual during settle
# ...
if bool(settling.any()):
    # DIRECT pre-position: drive cylinders to path_start direct_target
    q_start = scaling.clamp_usd_q(
        scaling.learned_x_to_usd_q(self._path_direct_targets[0].expand(len(ids), 4))
    )
    self._cylinder_pos_target[ids] = q_start
```

Settle is **hard-coded as 400 steps** (`residual_settle_steps` in env_cfg.py:37). At 1/60 s per step, that is **6.67 seconds**. During settle:
- PPO residual action is zeroed
- Cylinders are driven directly to `_path_direct_targets[0]` (a pre-computed PhysX target at the path start)
- At the handoff (step == settle), the controller state is reset to match measured joint positions

Settle is defined in **steps, not seconds**. Changing `step_dt` or `decimation` would change the settle duration unless `residual_settle_steps` is also adjusted.

## 6. Whether truck motion occurs continuously or in phases

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py:100-109`

```python
def _apply_truck_speed(self, speed_action: torch.Tensor) -> None:
    base = float(getattr(self.cfg, "residual_truck_base_kmh", 0.0))
    span = self._effective_truck_span()
    # ...
    self.robot.write_joint_velocity_to_sim(wheel, joint_ids=self._wheel_joint_ids)
```

Called every physics step from `_pre_physics_step`. Truck speed is **continuous** — the policy outputs a speed action at every step. During curriculum phase 0, `span=0` so the truck is parked (but still receives zero-velocity commands). During phase 1-2, the truck drives at whatever speed the policy commands.

The truck does **not** operate in discrete advance/brake phases. There is no parking brake, no brake testing, no stopped scoring. The truck either stays still (phase 0) or moves continuously (phase 1-2).

## 7. Whether PPO owns truck speed

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py:112-128`

Yes. `action[:, 2]` is the truck speed action in [-1, 1], mapped to km/h via:
```python
kmh = clamp(base + action[:, 2] * span, 0, max)
```

PPO fully controls truck speed. In phase 0, `span=0` so the action has no effect. In phases 1-2, PPO can command 0-4.5 (slow) or 0-6.0 (full) km/h. There is **no deterministic truck advance** — the PPO must learn to drive.

## 8. Which frame the nominal target uses

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py:43-49` and `ID_FD_pipline/tasks/cylinder_env.py` (`_compute_ee_pos_local`)

The goals (`path.goals`) are stored as `cutter_truck` positions — cutter coordinates in the **truck `base_link` frame**. The EE position is computed in `_compute_ee_pos_local()`:

```python
base_id = self._truck_base_body_id()
base_pos_w = self.robot.data.body_link_state_w[:, base_id, :3]
base_quat_w = self.robot.data.body_link_state_w[:, base_id, 3:7]
tcp_local = quat_apply_inverse(base_q, ee_pos_w - base_pos_w)
```

**CODE EVIDENCE:** The truck base body is `base_link` or `Frontrahmen` (not the articulation root `Drehzapfen`). From `cylinder_env.py`: "Reference frame = TRUCK CHASSIS (base_link), NOT root_pos_w. The articulation ROOT is the Drehzapfen -- the ARM'S OWN SLEW BASE."

This is correct. The goals and measurements are in the same truck-chassis frame.

## 9. Which physical point is controlled

**CODE EVIDENCE:** `ID_FD_pipline/tasks/cylinder_env.py:_compute_cutter_pos_w()` and `_compute_ee_pos_local()`

The controlled point is the **CUTTER**, 0.79 m below the `Messerkopf` body origin:

```python
offset = float(getattr(self.cfg, "tool_tip_local_offset_z", -0.79))
tip_body[:, 2] = offset
ee_pos_w = ee_pos_w + quat_apply(ee_q, tip_body)
```

All goals, Jacobians, and measurements use the cutter tip, not the body origin. The offset is applied in the live EE-body frame.

## 10. How the cutter offset is applied

**CODE EVIDENCE:** `ID_FD_pipline/tasks/cylinder_env.py:_compute_cutter_pos_w()`

The offset `(0, 0, -0.79)` is added in the **Messerkopf body-local frame**, then rotated to world by the body quaternion. This is correct for a rigid tool attached to the EE body.

## 11. How the EE target becomes cylinder targets

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py:120-125` and `ID_FD_pipline/dls_solver/ee_cascade.py:cylinder_goal_seeded()`

```python
# In env.py _pre_physics_step:
corrected, cylinder_goal = self.controller_targets(self._residual_action)
# -> calls ee_cascade.cylinder_goal_seeded(corrected_goal, nominal_seed, correction_gain=0.40)

# In ee_cascade.py cylinder_goal_seeded():
# seed is the primary IK solution from the ground map
# correction = gain * (goal - measured_ee) @ pinv.T  (bounded DLS correction)
# requested = clamp(seed + correction, learned_min, learned_max)
# -> bounded to max_cylinder_lead_m from measured state
```

The path provides both an EE goal AND a cylinder seed. The cascade uses the seed as the primary IK solution, adding a small measured-EE DLS correction. This is correct for staying on the mapped IK branch.

## 12. Which Jacobian is used

**CODE EVIDENCE:** `ID_FD_pipline/dls_solver/ee_cascade.py:34-39` and `ID_FD_pipline/ppo1/env.py:55-59`

The PPO env creates `EECascadeController` without passing a Jacobian, so it calls `ee_xyz_jacobian_stepped(env)[0]` once at construction to measure the **stepped-PhysX cutter Jacobian**. This is expensive but correct — it solves the closed-loop linkage constraint.

The Jacobian is cached in the cascade controller and refreshed at reset via `refresh_jacobian()`. It is NOT refreshed mid-episode.

**Jacobian properties (from cached JSON):**

| Property | Reset Jacobian | Ground Jacobian |
|---|---|---|
| Shape | 3×4 | 3×4 |
| Rank | 3 | 3 |
| SVs | [10.42, 5.89, 3.51] | [8.24, 5.69, 2.79] |
| Condition | 2.97 | 2.96 |
| J[:,:3] rank | — | 3 (cond 3.85) |

## 13. Whether cylinder 4 is used for position or held for head orientation

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py` does NOT reserve cylinder 4. The full 4×4 Jacobian and 4-D cylinder seeds are used. Cylinder 4 participates in EE position control.

**CONTRAST:** `stop_and_mow_gate.py` in `--arm_only` mode zeroes `ground_jacobian[:, 3] = 0.0` and uses `ground_pinv` from `J[:,:3]`. The README states: "cylinder 4 is held at the calibrated ground reference. Its measured travel was 0.3 mm."

## 14. Where ground-height correction is applied

**Nowhere in the PPO env.** The path goals come from `ground_band_map.json` samples where `|cutter_world_z| <= 0.10`, so they are already near ground level. But there is no active ground-height feedback correction.

**CONTRAST:** `stop_and_mow_gate.py` applies active clearance correction:
```python
clearance = float(env._compute_cutter_world_z()[0].item())
goal_np[2] += float(np.clip(0.05 - clearance, -0.12, 0.12))
```

## 15. Whether nominal references are being mutated by feedback

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py:120-125`

The corrected EE goal is stored (`self._corrected_ee_goal_local = corrected`) but the nominal goal (`self._ee_goal_local`) is computed fresh every step from the path. The nominal path is NOT mutated by feedback. The PPO residual is applied as an additive offset, and the corrected goal is what the cascade tracks. This separation is correct.

## 16. Which variables are used in PPO observations

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/observations.py:33` (33-D observation)

| Component | Dims | Source |
|---|---|---|
| EE position YZ | 2 | `_compute_ee_pos_local()` |
| Nominal goal YZ | 2 | `_ee_goal_local` from path |
| Error YZ | 2 | goal - position |
| EE velocity YZ | 2 | finite difference |
| Cylinder state X | 4 | `_refresh_cylinder_state()` |
| Cylinder velocity Xdot | 4 | `_refresh_cylinder_state()` |
| Vehicle speed / 10 | 1 | `_vehicle_speed_kmh` |
| Vehicle yaw rate | 1 | `root_ang_vel_w[:, 2]` |
| Previous residual action | 3 | `_previous_residual_action` |
| Cylinder goal | 4 | `_cylinder_goal_x` |
| Valve command | 4 | `_current_valve_cmd` |
| FD delta | 4 | `_fd_delta_corrected` |

Missing: phase information, path progress, ground height, head-down alignment, height sensors.

## 17. Which reward terms unintentionally encourage continuous driving

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/rewards.py` (DeepSeek Stage B)

```python
truck_credit = exp(-error / decay)   # decay = 0.10
truck_reward = truck_credit * (kmh / max_kmh)
progress_term = 1.5 * truck_reward
```

The exponential credit gives smooth, always-nonzero truck reward as long as tracking is reasonable. With decay=0.10, even 0.2 m error gives truck_credit = exp(-2) = 0.135, so the policy gets ~0.20 bonus for driving. This encourages the policy to drive whenever tracking is "good enough," which may conflict with stop-and-mow where the truck should be PARKED during mowing.

## 18. Whether zero action exactly reproduces the deterministic controller

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/tests/zero_action_contract.py`

This test asserts:
- Zero action produces `corrected == _ee_goal_local` (no modification)
- `cylinder_goal` from `controller_targets(zero)` matches `cylinder_goal_seeded(goal, seed)` 
- 16-step rollout with zero action has finite observations and rewards
- 30-step rollout with zero action: truck does NOT move (< 0.05 m displacement)
- Probe action [1.0, -1.0, 0.0] produces ±0.05 m Y/Z correction, X unchanged

**STATUS:** The test exists but has not been runtime-verified in this audit (requires Isaac Sim). The code logic is correct — zero residual action exactly passes through the nominal controller.

## 19. Whether any code path bypasses ID/FD with direct articulation control

**CODE EVIDENCE:** `ID_FD_pipline/ppo1/env.py:129-141` (settle phase)

During the settle phase, cylinder position targets are set directly from `_path_direct_targets[0]`:
```python
q_start = scaling.clamp_usd_q(
    scaling.learned_x_to_usd_q(self._path_direct_targets[0])
)
self._cylinder_pos_target[ids] = q_start
```

This bypasses the ID/FD pipeline during settle. The direct targets are pre-computed offline PhysX targets at the path start — they are the "known good" configuration. This is acceptable for pre-positioning but should not occur during scored mowing.

During scored motion (post-settle), the normal ID/FD pipeline runs: `_run_cylinder_id_pipeline()` is called, which runs the learned ID → valve → FD → delta → PhysX target chain.

## 20. Episode timeline simulation

**RUNTIME EVIDENCE (computed from config):**

| Phase | Steps | Duration | What happens |
|---|---|---|---|
| Settle | 0-399 | 6.67 s | Cylinders driven to path_start direct_target. Residual zeroed. Goal at path[0]. |
| Handoff | 400 | — | Controller reset to measured joint state. |
| Motion | 401-2099 | 28.33 s | Goal advances at 0.045 m/s along path. Arm tracks. |
| Total | 2100 | 35.00 s | |

Goal travel during motion: 28.33 × 0.045 = 1.27 m (34% of 3.8 m path).
Goal advance per step: 0.75 mm.

## Summary of critical findings

1. **Path is one-way forward arc, not back-and-forth mowing.** No return stroke, no cosine turnaround.
2. **Memory.md explicitly warns against using arc() as runtime IK.** "Do not reuse those raw graph branches as runtime IK seeds."
3. **Cylinder 4 is NOT reserved for head-down.** Full 4×4 Jacobian used.
4. **No ground-height feedback correction.** Path is assumed flat from map samples.
5. **Truck motion is continuous policy-controlled, not phased advance/brake.**
6. **Goal advances by discrete waypoint jumps** (2 cm every 0.44 s), not continuous interpolation.
7. **Settle is hard-coded 400 steps** (6.67 s), not duration-based.
8. **Truck progress reward encourages driving** during arm mowing, which conflicts with stop-and-mow.
9. **Observations lack phase, progress, and ground-height information.**
10. **Path uses graph-theoretic diameter through disconnected map**, not the verified 1.00 m lateral centred span.
