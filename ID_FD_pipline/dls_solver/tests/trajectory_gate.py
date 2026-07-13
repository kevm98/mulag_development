#!/usr/bin/env python3
"""Cascaded EE->cylinder->action DLS trajectory execution gate."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--warmup_steps", type=int, default=100)
p.add_argument("--vehicle_mode", choices=("stationary", "moving", "both"), default="both")
p.add_argument("--tolerance_m", type=float, default=0.05)
p.add_argument("--backend", choices=("idfd", "direct"), default="idfd")
p.add_argument("--shapes", default="line,curve,sweep")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.num_envs != 1:
    raise ValueError("Trajectory gate requires --num_envs 1")

random.seed(a.seed)
np.random.seed(a.seed)
app = AppLauncher(a)
sim_app = app.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(a.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(a.seed)

import ID_FD_pipline.tasks  # noqa: E402,F401
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController  # noqa: E402
from ID_FD_pipline.dls_solver.ee_cascade import EECascadeController  # noqa: E402
from ID_FD_pipline.dls_solver.kinematics import ee_xyz_jacobian_stepped  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


TASK = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=1)
cfg.seed = int(a.seed)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 300.0
cfg.ee_success_hold_steps = 100000
cfg.low_level_servo_mode = "direct_position_delta" if a.backend == "direct" else "none"
cfg.direct_position_delta_clip_m = 0.0015
# REALISTIC scale. The old values (speed 0.001, length 0.06, amp 0.025) made this
# gate VACUOUS: the goal travelled ~9.5 mm over the whole run -- less than the
# reported tracking error -- so a motionless arm "passed". Non-vacuity is now
# asserted explicitly below (goal must actually traverse a path, and the EE must
# actually follow it).
# Sized to the MEASURED reachable envelope (yz_workspace_probe.py):
# dy +-0.271 m, dz +-0.232 m about the anchor. The line/sweep are one-sided in Y
# (anchor -> anchor+length) and the curve reaches z = 2*radius, so these must fit.
cfg.ee_trajectory_speed_mps = 0.04
cfg.ee_trajectory_length_m = 0.35            # lateral mowing swath (Y)
cfg.ee_trajectory_radius_m = 0.20            # gentler 0.399 m arc; task length is unchanged
cfg.ee_trajectory_sweep_amp_m = 0.05         # small Z ripple (terrain following)
cfg.ee_trajectory_sweep_wavelength_m = 0.30

# --- Non-vacuity thresholds (a tracking gate is meaningless without these) ---
MIN_GOAL_PATH_M = 0.25      # do NOT lower this to make the gate pass
MIN_FOLLOW_RATIO = 0.60     # the EE must actually travel a comparable distance
MAX_ERR_FRAC_OF_SPAN = 0.25 # error must be small RELATIVE to the motion
env = gym.make(TASK, cfg=cfg).unwrapped


@torch.no_grad()
def settle(steps: int = 30) -> None:
    zero = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
    for _ in range(steps):
        env.step(zero)


def set_parking_brake(enabled: bool) -> None:
    wheel_ids = env._wheel_joint_ids
    env.robot.write_joint_stiffness_to_sim(6.0e8 if enabled else 0.0, joint_ids=wheel_ids)
    env.robot.write_joint_damping_to_sim(1.0e6, joint_ids=wheel_ids)
    if enabled:
        current = env.robot.data.joint_pos[:, wheel_ids].detach().clone()
        env.robot.set_joint_position_target(current, joint_ids=wheel_ids)




@torch.no_grad()
def run_shape(mode: str, moving: bool, jacobian: torch.Tensor) -> dict[str, float | bool]:
    print(f"[EE-TRAJ][ENTER] mode={mode} moving={moving}", flush=True)
    env.cfg.enable_scripted_vehicle_motion = False
    env.reset(seed=int(a.seed))
    set_parking_brake(True)
    print("[EE-TRAJ][ENTER] reset done", flush=True)
    settle()
    print("[EE-TRAJ][ENTER] settle done", flush=True)
    controller = CylinderDLSController(
        env,
        CylinderDLSConfig(
            invert_learned_pipeline=False,
            error_gain=0.08 if mode == "sweep" else 0.10,
            max_desired_delta_m=0.01,
            max_action=1.0,
            action_smoothing=1.0 if moving else 0.35,
        ),
    )
    print("[EE-TRAJ][ENTER] controller built", flush=True)
    anchor = env._compute_ee_pos_local().detach().clone()
    drop = max(float(env._compute_cutter_world_z()[0].item()) - 0.05, 0.0)
    anchor[:, 2] -= drop
    env._traj_mode = mode
    env._traj_anchor[:] = anchor
    env._traj_u[:] = torch.tensor([0.0, 1.0, 0.0], device=env.device)
    env._traj_v[:] = torch.tensor([1.0, 0.0, 0.0], device=env.device)
    env.episode_length_buf.zero_()
    env._update_ee_trajectory_goal()
    print(f"[EE-TRAJ][ENTER] standard DLS {mode} path", flush=True)

    # Use the SHARED cascade (resolved-rate: pose-adaptive Jacobian + velocity
    # feedforward). Previously this test reimplemented a fixed-anchor-Jacobian law
    # inline, which is exactly what undershot the path.
    cascade = EECascadeController(
        env,
        cylinder_controller=controller,
        jacobian=jacobian,
        ee_feedback_gain=0.10,
        max_cylinder_lead_m=0.01,
    )
    print("[EE-TRAJ][ENTER] cascade built", flush=True)
    cascade.reset()
    print("[EE-TRAJ][ENTER] cascade reset", flush=True)
    start_goal = env._traj_anchor.detach().clone()
    for _ in range(250):
        controller.step(cascade.cylinder_goal(start_goal))
    env.episode_length_buf.zero_()
    env._update_ee_trajectory_goal()
    set_parking_brake(not moving)
    env.cfg.enable_scripted_vehicle_motion = bool(moving)
    if moving:
        for _ in range(200):
            controller.step(cascade.cylinder_goal(start_goal))
        controller.reset(); cascade.reset()
    env.episode_length_buf.zero_()
    env._update_ee_trajectory_goal()
    print("[EE-TRAJ][ENTER] pre-positioned on ground path", flush=True)
    print(f"[EE-TRAJ][JAC] mode={mode} truck-frame 3x4 physx={cascade.jacobian.cpu().tolist()}", flush=True)

    truck_body_id = env._truck_base_body_id()
    truck0 = env.robot.data.body_link_state_w[0, truck_body_id, :3].detach().clone()
    cyl_hist: list[np.ndarray] = []
    errors: list[float] = []
    track_err: list[float] = []
    ik_err: list[float] = []
    goal_yz_hist: list[np.ndarray] = []
    ee_yz_hist: list[np.ndarray] = []
    cutter_z_hist: list[float] = []
    for step in range(int(a.steps)):
        goal = env._ee_goal_local.detach().clone()
        cylinder_goal = cascade.cylinder_goal(goal)
        controller.step(cylinder_goal)
        ee_after = env._compute_ee_pos_local()
        x_meas, _ = env._refresh_cylinder_state()
        track_err.append(float(torch.linalg.norm(cylinder_goal[0] - x_meas[0]).item()))
        ik_err.append(float(torch.linalg.norm(cascade._ee_ref[0] - goal[0, :3]).item()))
        error = float(torch.linalg.norm(goal[0] - ee_after[0, :3]).item())
        errors.append(error)
        cutter_z_hist.append(float(env._compute_cutter_world_z()[0].item()))
        cyl_hist.append(env._refresh_cylinder_state()[0][0].detach().cpu().numpy().copy())
        goal_yz_hist.append(goal[0].detach().cpu().numpy().copy())
        ee_yz_hist.append(ee_after[0, :3].detach().cpu().numpy().copy())
        if step % 200 == 0:
            print(
                f"[EE-TRAJ] vehicle={'moving' if moving else 'stationary'} mode={mode} "
                f"step={step:04d} yz_error={error:.4f}",
                flush=True,
            )

    # --- NON-VACUITY: did the goal actually move, and did the EE actually follow? ---
    G = np.asarray(goal_yz_hist); E = np.asarray(ee_yz_hist)
    goal_path = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
    ee_path = float(np.linalg.norm(np.diff(E, axis=0), axis=1).sum())
    goal_span = float(np.linalg.norm(G.max(0) - G.min(0)))
    ee_span = float(np.linalg.norm(E.max(0) - E.min(0)))
    follow_ratio = ee_path / max(goal_path, 1e-9)

    steady = np.asarray(errors[max(int(a.warmup_steps), 0) :], dtype=np.float64)
    tail = steady[-200:] if len(steady) >= 200 else steady
    previous = steady[-400:-200] if len(steady) >= 400 else steady[: len(tail)]
    mean = float(np.mean(steady))
    p95 = float(np.percentile(steady, 95.0))
    final = float(np.mean(tail[-50:]))
    oscillation = float(np.std(tail))
    drift = float(abs(np.mean(tail) - np.mean(previous))) if len(previous) else 0.0
    tracking_ok = p95 < a.tolerance_m and final < a.tolerance_m and oscillation < 0.02 and drift < 0.015
    # A tracking gate is meaningless unless the task was non-trivial:
    nonvacuous = (
        goal_path >= MIN_GOAL_PATH_M
        and follow_ratio >= MIN_FOLLOW_RATIO
        and mean <= MAX_ERR_FRAC_OF_SPAN * max(goal_span, 1e-9)
    )
    Z = np.asarray(cutter_z_hist, dtype=np.float64)
    ground_fraction = float(np.mean(np.abs(Z) <= 0.10))
    ground_ok = ground_fraction >= 0.90
    passed = bool(tracking_ok and nonvacuous and ground_ok)
    print(
        f"[EE-TRAJ][RESULT] vehicle={'moving' if moving else 'stationary'} mode={mode} "
        f"{'PASS' if passed else 'FAIL'} mean={mean:.4f} p95={p95:.4f} final={final:.4f} "
        f"tail_std={oscillation:.4f} drift={drift:.4f}",
        flush=True,
    )
    C = np.asarray(cyl_hist)
    cyl_range = (C.max(0) - C.min(0))
    print(f"[EE-TRAJ][ARM-MOTION] cylinder travel per cyl = {np.round(cyl_range,4).tolist()} m "
          f"| max={cyl_range.max():.4f} -> {'ARM IS ARTICULATING' if cyl_range.max() > 0.03 else '**ARM IS NOT MOVING**'}", flush=True)
    print(f"[EE-TRAJ][CUTTER] ground_fraction={ground_fraction:.3f} "
          f"world_z=[{Z.min():+.3f},{Z.max():+.3f}] m "
          f"-> {'MOWING AT GROUND' if ground_ok else '**NOT AT GROUND**'}", flush=True)
    truck_moved = float(torch.linalg.norm(
        env.robot.data.body_link_state_w[0, truck_body_id, :3] - truck0
    ).item())
    print(
        f"[EE-TRAJ][TRUCK] moving={moving} truck_displacement={truck_moved:.3f} m "
        f"speed_kmh={float(env._vehicle_speed_kmh[0].item()):.2f} "
        f"-> {'TRUCK MOVED' if truck_moved > 0.5 else '**TRUCK DID NOT MOVE**'}",
        flush=True,
    )
    print(
        f"[EE-TRAJ][DIAG] mean|cyl_ref-cyl_meas|={np.mean(track_err):.4f}m  "
        f"mean|FK(ref)-goal|={np.mean(ik_err):.4f}m   "
        f"-> {'CYLINDER CTRL not tracking' if np.mean(track_err) > 0.02 else 'cyl ctrl OK'}; "
        f"{'IK not converging' if np.mean(ik_err) > 0.02 else 'IK OK'}",
        flush=True,
    )
    print(
        f"[EE-TRAJ][NONVACUITY] goal_path={goal_path:.3f}m ee_path={ee_path:.3f}m "
        f"follow_ratio={follow_ratio:.2f} goal_span={goal_span:.3f}m ee_span={ee_span:.3f}m "
        f"err/span={mean / max(goal_span, 1e-9):.2f} -> {'OK' if nonvacuous else '**VACUOUS**'}",
        flush=True,
    )
    return {
        "passed": passed, "mean": mean, "p95": p95, "final": final,
        "goal_path": goal_path, "ee_path": ee_path, "follow_ratio": follow_ratio,
        "nonvacuous": nonvacuous,
    }


try:
    env.reset(seed=int(a.seed))
    settle()
    _cache = Path(__file__).resolve().parents[2] / "models/ee_jacobian_ground_truck3x4.json"
    if _cache.is_file():
        import json as _json
        J0 = torch.tensor(_json.loads(_cache.read_text())["jacobian"], device=env.device)
        print(f"[EE-TRAJ] loaded cached truck-frame jacobian from {_cache.name}", flush=True)
    else:
        J0 = ee_xyz_jacobian_stepped(env)[0]  # WARNING: Kit may abort on later resets
    sv = torch.linalg.svdvals(J0.cpu())
    print(f"[EE-TRAJ] truck-frame 3x4 Jacobian rank={int((sv > 0.01 * sv[0]).sum())}/3 "
          f"SVs={[f'{float(v):.4e}' for v in sv]}", flush=True)
    vehicle_states = [False, True] if a.vehicle_mode == "both" else [a.vehicle_mode == "moving"]
    results = []
    shapes = tuple(value.strip() for value in a.shapes.split(",") if value.strip())
    if not shapes or any(value not in ("line", "curve", "sweep") for value in shapes):
        raise ValueError("--shapes must contain line, curve, and/or sweep")
    for moving in vehicle_states:
        for mode in shapes:
            results.append(run_shape(mode, moving, J0))
    passed = all(bool(result["passed"]) for result in results)
    print(f"[EE-TRAJ][GATE] {'PASS' if passed else 'FAIL'}", flush=True)
finally:
    env.close()
    sim_app.close()
