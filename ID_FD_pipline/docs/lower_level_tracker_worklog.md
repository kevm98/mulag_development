# Lower-level tracker worklog

Branch `fix/lower-level-idfd-tracker` off `93b54f3`.
Measurements and conclusions live in `lower_level_tracker_results.md`; this file is the
chronological log of what was run and why.

## Step 1 — inspect
Read `ppo1/env.py`, `dls_solver/{cylinder,ee_cascade,kinematics}.py`,
`trajectories/{arc_mowing,sampler,time_scaling}.py`, `tasks/{env_cfg,cylinder_env}.py`,
`plant_study/direct_drive.py`, and the ID→FD pipeline core in
`kevin_integration/.../mulag_upper_level_env.py`.

Active configuration confirmed: `low_level_servo_mode="none"` ⇒ `CylinderDLSController`
takes the composite local-inversion branch (option A). `paired_id_fd_residual=True`,
`id_residual_over_zero_target=True`, direction guard on with ratio 1.5,
`cylinder_target_integrates_from_previous_target=True`, `leak=0`, `action_scale=0.02`,
`step_dt=1/60 s`.

First smell, from `models/ground_band_map.json`: commanded `target_x` and settled
`cylinder_x` differ by up to **0.25 m**. A position drive should not have a 0.25 m
steady-state offset.

## Step 2 — bench
Built `ppo1/tests/lower_level_bench.py` on the plain cylinder-space env (4-D action ==
cylinder DLS action). No state machine, no EE IK, no truck motion, PPO residual absent by
construction. Modes: `truck`, `hold`, `cyl`, `jac`, `plant`.

## Step 3 — static hold at endpoint A
`--mode hold` with the two possible handoffs.
* handoff=target (sync PhysX target to the *commanded* value): EE p95 **2.5 mm**.
* handoff=measured (what production does): EE p95 **61 mm**, 7.6 cm sag transient.

Hypothesis: production's `_cylinder_pos_target = measured` at `ppo1/env.py:438` throws away
the command the plant was in equilibrium under. Confirmed.

But `target − measured` settled at **−0.165 m** on cylinder 1 while the arm sat perfectly
still. A position drive holding 16.5 cm of error without moving is not a controller
result — it is a stuck plant. Flagged as suspicious rather than accepted as a pass.

## Step 4 — per-cylinder matrix
`--mode cyl` at endpoint A: every ±5/±10 mm step ends with the error **equal to the full
step amplitude** — cylinders 1 and 3 do not move at all. `fd_vs_actual_gain` ≈ 0.00–0.20.

## Step 5 — plant probe (the answer)
`--mode plant`: DirectCylinderDriver alone, no ID/FD/DLS/guards. `d(measured)/d(target) ≈
0` for all four cylinders in both directions out to ±50 mm. `applied_torque ==
effort_limit` on all four; `computed_torque` on cylinder 1 is **1.46e7 vs a 5e4 limit
(293×)**.

Effort-limit sweep: at 16× effort the arm actually executes the commanded endpoint-A
cylinder values and the cutter lands at (2.28, −2.24, 0.78) — **>3 m from where the band map
says endpoint A is**. So the band map, the arc fit and the ground Jacobian are all built on
saturated sag poses.

## Step 6 — reset-pose control test
`--mode cyl --pose reset` (a pose where the drives are near-linear and the plant does move,
235–1275 µm/step). ID→FD now moves the arm but **overshoots 5–8× and limit-cycles**: p95
15–107 mm on ±5–10 mm steps. The DLS inverts FD's predicted delta, not the plant's actual
response to the integrated target; measured loop gain is ~2.6× what it assumes.

## Status
Halted before controller redesign. Bottleneck #1 (actuator effort/stiffness) and #2
(sag-derived arc + band map + Jacobian) are plant/artifact problems that sit upstream of
the lower-level tracker and outside the "do not modify initially" list. Tuning gains,
adding velocity feedforward, building a cylinder teacher table or fine-tuning ID/FD against
this plant would all be work spent on a plant that cannot move. Escalated for a decision.

## Not done (deliberately)
* 50 s zero-action run, 60 s truck-freeze proof — not on the critical path while §2 stands.
* Composite IDFD Jacobian at A/mid/B — a Jacobian measured where targets do nothing is noise.
* Cruise/S-curve time profile, velocity feedforward, teacher table, guard ablations,
  task-space gates, stress matrix — all blocked on a controllable plant.
