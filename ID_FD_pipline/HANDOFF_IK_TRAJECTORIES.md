# HANDOFF: IK-planned long ground-level trajectories (+ visual verification)

## READ FIRST — saved lessons (do NOT repeat these mistakes)
Read `/home/rrlab/.claude/projects/-home-rrlab-rrlab-mulag/memory/idfd-mowing-hard-won-lessons.md`
(all of it, especially sections 1, 2, 6). Hard rules distilled:

1. **NEVER shrink a task to make a gate pass.** Non-vacuity floors are sacred:
   `goal_path >= 0.25 m` (for the NEW long-line gate: `>= 1.0 m`), `ee_path/goal_path >= 0.6`,
   `mean_error <= 0.25 * goal_span`, and always print cutter world-z, per-cylinder travel,
   truck displacement. The user verifies VISUALLY — a motionless arm with green metrics is a bug.
2. **This arm is a CLOSED-LOOP linkage** (passive `*_constraint_joint_*` DOFs). Any kinematics
   via `write_joint_state_to_sim` + `sim.forward()` is INVALID (welds the loop). The native
   `root_physx_view.get_jacobians()` sliced to cylinder columns is EQUALLY invalid (third SV
   ~1e-7 artifact). The ONLY valid kinematics: command cylinder position targets and STEP
   physics (`plant_study/direct_drive.py::DirectCylinderDriver`, `ee_xyz_jacobian_stepped`).
3. **Reference frame = truck chassis (`base_link` body), never `root_pos_w/root_quat_w`** —
   the articulation root is the `Drehzapfen` (arm slew base) and rotates with the arm.
4. **The controlled point is the CUTTER**, 0.79 m below the `Messerkopf` body origin
   (`tool_tip_local_offset_z = -0.79`, applied in `ID_FD_pipline/tasks/cylinder_env.py::_compute_ee_pos_local`).
5. **Kit swallows exceptions and exits 0** (even NameError; also CUDA cuSOLVER SVD aborts).
   Do small SVDs on CPU. `python -m py_compile` does not catch stale names — smoke-run every
   edited entry point and bisect with prints when a run dies silently.
6. Stepped-Jacobian rollouts followed by more in-process work can abort Kit — measure in a
   **one-shot process** and cache (`plant_study/measure_ee_jacobian.py` →
   `ID_FD_pipline/models/ee_jacobian_truck3x4.json`).
7. FD/ID checkpoints are LOCKED (causally calibrated; cylinder DLS tracks 1.3–1.5 cm). No RL
   work: the old `ppo1` residual env assumes the rotating Drehzapfen frame and is INVALID —
   leave it alone until the controller stack is final.

## Current verified state (baseline you must not regress)
`./rrlab.sh -p ID_FD_pipline/dls_solver/tests/trajectory_gate.py --backend idfd --vehicle_mode moving --warmup_steps 200 --headless`
→ **GATE PASS**: line 3.6 / curve 3.8 / sweep 3.7 cm mean (p95 < 4.9 cm), cutter world
z = +0.02..0.03 m throughout, truck driving, arm articulating.
Passing config inside the gate: truck-frame goals, cached stepped Jacobian (fixed per run),
`CylinderDLSConfig(error_gain=0.10 (0.08 sweep), max_desired_delta_m=0.01, max_action=1.0,
action_smoothing=0.35)`, unscored 250-step pre-position descent, warmup 200/600.

## Current structure
```
ID_FD_pipline/
  tasks/cylinder_env.py      truck-frame cutter EE (_compute_ee_pos_local, _truck_base_body_id),
                             parametric traj generator (_update_ee_trajectory_goal: line/curve/sweep)
  tasks/env_cfg.py           wheels switched to velocity drive (stiffness 0, damping 1e6)
  dls_solver/kinematics.py   ee_xyz_jacobian_stepped (VALID) + analytic helpers (INVALID for control)
  dls_solver/ee_cascade.py   3-D truck-frame IK cascade (reference-space Newton, CPU SVD)
  dls_solver/cylinder.py     4x4 cylinder DLS (closes on MEASURED state)
  dls_solver/tests/trajectory_gate.py  the passing gate (non-vacuity + diagnostics)
  models/ee_jacobian_truck3x4.json     cached stepped Jacobian (rank 3/3, SVs ~[10.9,5.9,3.3])
plant_study/                 direct_drive.py, measure_ee_jacobian.py, physics probes
```

## Known issues that may need changing (and why)
- The `-0.79` cutter offset is duplicated (env + gate prints). Centralize on
  `cfg.tool_tip_local_offset_z`; read it everywhere instead of literals.
- The trajectory generator only does parametric shapes anchored at the current pose. It has
  **no waypoint-path mode** — that is what you are adding (below).
- The ground anchor currently sits at the **edge** of the lateral ground band
  (`ground_reach_probe`: ≥1.56 m reach one-sided, ~0.08 m the other) — long paths must be
  planned from the band's geometry, not from the arrival pose.
- Truck-speed RNG depends on shape order (env RNG is not reseeded per shape) — reseed
  `torch`/env RNG per shape if you need per-shape reproducibility.
- A parked truck rolls ~0.35 m from arm reaction forces (wheels are velocity-drive). If
  stationary tests matter, add a parking mode (restore high stiffness when not driving).

## THE TASK: plan long trajectories by inverse kinematics over the ground-level workspace

### Step 1 — Ground-band workspace map (offline, physics-settled)
New: `plant_study/map_ground_band.py`. Grid the 4 cylinder extensions over their learned
ranges (start 6^4 = 1296 poses; subsample if slow), drive each pose with
`DirectCylinderDriver.step()` until settled (~40 steps), record
`(cylinder_x[4], cutter_pos_truck_frame[3], cutter_world_z)`. Keep poses with
`|cutter_world_z| <= 0.10` → save ALL samples + the ground-band subset to
`ID_FD_pipline/models/ground_band_map.json`. Print the band's truck-frame extent
(x/y min/max, arc length). One-shot process; close the app when done (lesson 6).

### Step 2 — IK path planning on the map
New: `ID_FD_pipline/dls_solver/ground_paths.py`. From the map's ground band:
- Fit the **slew arc** (the metres-long natural path) ordered by arc length; also support a
  straight lateral line and a serpentine sweep confined to the band.
- Emit references as `(s -> cutter_goal_truck[3], cylinder_seed[4])` — every waypoint carries
  the mapped cylinder configuration, so feasibility holds **by construction** and the cascade
  IK can be seeded/`_cyl_ref`-warm-started along the branch.
- Target path lengths: line/arc **>= 1.5 m** (probe showed >= 1.56 m one-sided), speed
  0.04–0.06 m/s, so use ~1800 steps (30 s) runs.

### Step 3 — Long-trajectory gate
New: `ID_FD_pipline/dls_solver/tests/long_trajectory_gate.py` (copy the passing gate's
structure and diagnostics; do NOT weaken thresholds): non-vacuity `goal_path >= 1.0 m`,
cutter `|world z| <= 0.10 m` for >= 90% of scored steps, per-cylinder travel print,
truck modes stationary AND moving. Pre-position (unscored) to the path start via its
cylinder seed. tolerance_m stays 0.05.

### Step 4 — VISUAL verification commands (must work — the user will watch)
Provide and TEST a viewer entry point. IsaacLab-standard: the same script WITHOUT
`--headless` opens the Isaac Sim window (AppLauncher handles it; no extra flags needed).
Add `--realtime` sleep pacing if the sim runs faster than wall clock. Deliver exactly:
```bash
# watch the long arc with the truck driving (window opens; close it to stop)
./rrlab.sh -p ID_FD_pipline/dls_solver/tests/long_trajectory_gate.py --backend idfd --vehicle_mode moving --shapes arc
# watch the existing short-shape gate
./rrlab.sh -p ID_FD_pipline/dls_solver/tests/trajectory_gate.py --backend idfd --vehicle_mode moving --shapes sweep
```
Also draw the reference path in the viewer using
`isaaclab.markers.VisualizationMarkers` (standard IsaacLab; see
`standalone/tutorials/controllers/run_diff_ik.py` for the canonical pattern) so the user can
SEE goal vs cutter. Verify the non-headless run launches before handing over.

### Step 5 — Acceptance (all required)
1. `trajectory_gate.py` (short shapes) still **PASS** — no regression.
2. `long_trajectory_gate.py` **PASS** on the arc (>= 1.5 m) stationary AND moving.
3. Viewer commands verified to open and show the arm sweeping ON THE GROUND along a
   visible marker path, truck driving.
4. No lowered thresholds anywhere (diff-check the gates).
5. Update `ID_FD_pipline/README.md` run/visualize section; note stale sections rather than
   letting them contradict reality.

Work in small verified steps; after each edited entry point do a smoke run (Kit exit-0
silent-death trap). Budget runs: prefer `--shapes` single-shape and stationary mode while
iterating; full both-modes gate only at the end.
