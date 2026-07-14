#!/usr/bin/env python3
"""Stationary-truck, ground-level A->B->A cutter-arc tracking gate.

The reference is the longest natural slew arc that the repaired plant can execute and hold.
It is loaded from ``models/mowing_arc_head_down.json``, whose teacher poses were measured with
DirectCylinderDriver and filtered for reachability, hold drift, saturation, and ground height.

Control remains the production lower stack:

    tool reference -> mapped cylinder seed + small EE DLS correction
                   -> cylinder DLS -> paired ID/strict-FD hybrid -> PhysX target drives

The truck drive is disabled and its initial wheel positions are held for the entire test.
Planar chassis drift is scored separately from physical suspension motion. In GUI
mode, green markers show the reference arc, yellow/magenta are A/B, red is the live command,
and blue is the measured cutter point.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--seed", type=int, default=17)
p.add_argument(
    "--arc", default=str(ROOT / "ID_FD_pipline/models/mowing_arc_head_down.json")
)
p.add_argument("--speed_mps", type=float, default=0.13)
p.add_argument("--accel_time_s", type=float, default=1.0)
p.add_argument("--seed_lead_time_s", type=float, default=0.25)
p.add_argument("--cycles", type=int, default=1, help="Number of complete A->B->A cycles")
p.add_argument("--tracking_tolerance_m", type=float, default=0.05)
p.add_argument(
    "--ground_tolerance_m", type=float, default=0.10,
    help="Blade-bottom ground band (verified teacher envelope plus tracking tolerance)",
)
p.add_argument("--truck_drift_tolerance_m", type=float, default=0.02)
p.add_argument("--ee_correction_gain", type=float, default=0.0)
p.add_argument("--head_down_min_alignment", type=float, default=0.95)
p.add_argument("--preposition_steps", type=int, default=300)
p.add_argument("--settle_steps", type=int, default=120)
p.add_argument("--realtime", action="store_true")
p.add_argument("--out", default="outputs/ground_arc_tracking/result.json")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.num_envs != 1:
    raise ValueError("Ground arc gate requires exactly one environment")
if a.speed_mps <= 0.0 or a.cycles < 1:
    raise ValueError("--speed_mps must be positive and --cycles must be >= 1")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import SPHERE_MARKER_CFG  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import ID_FD_pipline.tasks  # noqa: E402,F401
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController  # noqa: E402
from ID_FD_pipline.dls_solver.ee_cascade import EECascadeController  # noqa: E402
from ID_FD_pipline.trajectories.time_scaling import cruise_profile  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402

torch.manual_seed(int(a.seed))
np.random.seed(int(a.seed))

TASK = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=1)
cfg.seed = int(a.seed)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 100000.0
cfg.ee_success_hold_steps = 100000
cfg.low_level_servo_mode = "none"
cfg.sim.physx.enable_external_forces_every_iteration = True
cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 2
env = gym.make(TASK, cfg=cfg).unwrapped


payload = json.loads(Path(a.arc).expanduser().resolve().read_text())
teacher = payload["cylinder_teacher"]
if len(teacher) < 8:
    raise RuntimeError("head-down mowing arc does not contain a dense, verified teacher path")

theta = np.asarray([row["theta_deg"] for row in teacher], dtype=np.float64)
goals = np.asarray([row["cutter_truck"] for row in teacher], dtype=np.float64)
seeds = np.asarray([row["cylinder_cmd"] for row in teacher], dtype=np.float64)
world_z_teacher = np.asarray([row["cutter_world_z"] for row in teacher], dtype=np.float64)
order = np.argsort(theta)
theta, goals, seeds, world_z_teacher = (
    theta[order], goals[order], seeds[order], world_z_teacher[order]
)
teacher_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(goals, axis=0), axis=1))]
arc_length = float(teacher_s[-1])
if arc_length < 3.0:
    raise RuntimeError(f"Verified ground arc is only {arc_length:.3f} m; expected a long sweep")


def interpolate(distance_m: float) -> tuple[np.ndarray, np.ndarray, float]:
    s = float(np.clip(distance_m, 0.0, arc_length))
    goal = np.asarray([np.interp(s, teacher_s, goals[:, i]) for i in range(3)], dtype=np.float32)
    seed = np.asarray([np.interp(s, teacher_s, seeds[:, i]) for i in range(4)], dtype=np.float32)
    wz = float(np.interp(s, teacher_s, world_z_teacher))
    return goal, seed, wz


def truck_to_world(points: torch.Tensor) -> torch.Tensor:
    base = env.robot.data.body_link_state_w[0, env._truck_base_body_id()]
    q = base[3:7] / base[3:7].norm().clamp(min=1.0e-8)
    return base[:3] + quat_apply(q.reshape(1, 4).expand(len(points), -1), points)


def mower_down_alignment() -> float:
    """Alignment of the Messerkopf authored local +Z axis with world down."""
    q = env.robot.data.body_link_state_w[0, env._ee_body_id, 3:7]
    q = q / q.norm().clamp(min=1.0e-8)
    axis_w = quat_apply(
        q.reshape(1, 4), torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
    )[0]
    return float((-axis_w[2]).item())


def marker(prim_path: str, radius: float, color: tuple[float, float, float]):
    marker_cfg = SPHERE_MARKER_CFG.copy()
    marker_cfg.markers["sphere"].radius = radius
    marker_cfg.markers["sphere"].visual_material.diffuse_color = color
    return VisualizationMarkers(marker_cfg.replace(prim_path=prim_path))


def make_markers():
    if bool(a.headless):
        return None
    markers = (
        marker("/Visuals/GroundArc/Path", 0.035, (0.1, 0.9, 0.2)),
        marker("/Visuals/GroundArc/Goal", 0.09, (1.0, 0.1, 0.1)),
        marker("/Visuals/GroundArc/Cutter", 0.075, (0.1, 0.4, 1.0)),
        marker("/Visuals/GroundArc/A", 0.11, (1.0, 0.8, 0.05)),
        marker("/Visuals/GroundArc/B", 0.11, (0.9, 0.1, 0.9)),
    )
    base = env.robot.data.body_link_state_w[0, env._truck_base_body_id(), :3]
    env.sim.set_camera_view(
        (base + torch.tensor([10.5, -12.0, 7.0], device=env.device)).cpu().tolist(),
        (base + torch.tensor([5.0, -2.2, 1.0], device=env.device)).cpu().tolist(),
    )
    return markers


@torch.no_grad()
def show_markers(markers, dense_path: torch.Tensor, goal: torch.Tensor) -> None:
    if markers is None:
        return
    path_marker, goal_marker, cutter_marker, a_marker, b_marker = markers
    path_marker.visualize(truck_to_world(dense_path))
    goal_marker.visualize(truck_to_world(goal.reshape(1, 3)))
    cutter_marker.visualize(env._compute_cutter_pos_w())
    a_marker.visualize(truck_to_world(dense_path[0:1]))
    b_marker.visualize(truck_to_world(dense_path[-1:]))


@torch.no_grad()
def hold_wheels(wheel_target: torch.Tensor) -> None:
    ids = env._wheel_joint_ids
    env.robot.write_joint_stiffness_to_sim(2.0e9, joint_ids=ids)
    env.robot.write_joint_damping_to_sim(1.0e7, joint_ids=ids)
    env.robot.set_joint_position_target(wheel_target, joint_ids=ids)
    env.robot.set_joint_velocity_target(torch.zeros_like(wheel_target), joint_ids=ids)


@torch.no_grad()
def constrain_truck_xy(anchor_xy: torch.Tensor) -> None:
    """Cancel planar chassis drift without constraining arm rotation or suspension."""
    base_xy = env.robot.data.body_link_state_w[:, env._truck_base_body_id(), :2]
    correction = anchor_xy - base_xy
    if float(torch.linalg.norm(correction, dim=1).max().item()) <= 0.001:
        return
    root_pose = env.robot.data.root_state_w[:, :7].detach().clone()
    root_pose[:, :2] += correction
    env.robot.write_root_pose_to_sim(root_pose)


def main() -> None:
    global goals, seeds, world_z_teacher, teacher_s, arc_length
    env.reset(seed=int(a.seed))
    wheel_target = env.robot.data.joint_pos[:, env._wheel_joint_ids].detach().clone()
    hold_wheels(wheel_target)
    truck_before_deploy = env.robot.data.body_link_state_w[
        0, env._truck_base_body_id(), :3
    ].detach().clone()

    # Start at whichever verified endpoint is closest to reset. Reversing the teacher preserves
    # the same A<->B geometry and avoids an unsafe long deployment sweep before scoring.
    driver = DirectCylinderDriver(env)
    reset_x = driver.measured_x()[0].detach().cpu().numpy()
    if np.linalg.norm(reset_x - seeds[-1]) < np.linalg.norm(reset_x - seeds[0]):
        goals = goals[::-1].copy()
        seeds = seeds[::-1].copy()
        world_z_teacher = world_z_teacher[::-1].copy()
        teacher_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(goals, axis=0), axis=1))]
        arc_length = float(teacher_s[-1])
        print("[GROUND-ARC] reversed teacher so A is the endpoint closest to reset", flush=True)

    # Cosine-ramp to A while holding the original wheel positions. An instantaneous target
    # jump produced a 1.89 m truck impulse and is not a valid parked-arm deployment.
    target_a = torch.as_tensor(seeds[0], device=env.device, dtype=torch.float32).reshape(1, 4)
    start_x = driver.measured_x().clone()
    for step in range(int(a.preposition_steps)):
        u = float(step + 1) / max(int(a.preposition_steps), 1)
        blend = 0.5 * (1.0 - math.cos(math.pi * u))
        deploy_target = start_x + blend * (target_a - start_x)
        hold_wheels(wheel_target)
        driver.step(deploy_target, substeps=2)

    truck_after_deploy = env.robot.data.body_link_state_w[
        0, env._truck_base_body_id(), :3
    ].detach().clone()
    deployment_drift = float(torch.linalg.norm(truck_after_deploy - truck_before_deploy).item())
    print(f"[GROUND-ARC] deployment truck drift={deployment_drift:.4f}m", flush=True)

    scaling = env._cylinder_adapter.scaling
    q_a = scaling.clamp_usd_q(scaling.learned_x_to_usd_q(target_a))
    env._cylinder_pos_target[:] = q_a
    env._neutral_hold_cylinder_pos_target[:] = q_a
    env._prev_valve_cmd.zero_()
    env._current_valve_cmd.zero_()
    env._fd_delta_raw.zero_()
    env._fd_delta_corrected.zero_()
    env._prev_action.zero_()
    env._current_action.zero_()
    env._cylinder_id.reset()
    env._cylinder_fd.reset()

    ground_j = torch.tensor(
        json.loads((ROOT / "ID_FD_pipline/models/ee_jacobian_ground_truck3x4.json").read_text())["jacobian"],
        device=env.device,
        dtype=torch.float32,
    )
    # Cylinder 4 owns head pitch on this teacher branch; keep it at its mapped value.
    ground_j[:, 3] = 0.0
    cylinder = CylinderDLSController(
        env,
        CylinderDLSConfig(
            invert_learned_pipeline=True,
            error_gain=0.08,
            velocity_damping=0.0,
            velocity_feedforward=0.0,
            max_desired_delta_m=0.004,
            max_action=1.0,
            action_smoothing=0.5,
        ),
    )
    cascade = EECascadeController(
        env,
        cylinder_controller=cylinder,
        jacobian=ground_j,
        ee_feedback_gain=0.0,
        max_ik_step_m=0.004,
        max_cylinder_lead_m=0.025,
    )
    cascade.reset()

    for _ in range(int(a.settle_steps)):
        hold_wheels(wheel_target)
        cylinder.step(target_a)

    truck_start = env.robot.data.body_link_state_w[
        0, env._truck_base_body_id(), :3
    ].detach().clone()
    truck_anchor_xy = truck_start[:2].reshape(1, 2)

    dense_s = np.linspace(0.0, arc_length, 220)
    dense_path = torch.as_tensor(
        np.asarray([interpolate(s)[0] for s in dense_s]), device=env.device, dtype=torch.float32
    )
    markers = make_markers()

    zero_t = torch.zeros(1, device=env.device)
    _s0, _v0, _a0, one_way_s = cruise_profile(
        zero_t,
        length_m=arc_length,
        cruise_speed_mps=float(a.speed_mps),
        accel_time_s=float(a.accel_time_s),
        decel_time_s=float(a.accel_time_s),
    )
    total_s = 2.0 * one_way_s * int(a.cycles)
    steps = int(math.ceil(total_s / float(env.step_dt))) + 1
    print(
        f"[GROUND-ARC] A={np.round(goals[0],3).tolist()} B={np.round(goals[-1],3).tolist()} "
        f"arc={arc_length:.3f}m chord={float(np.linalg.norm(goals[-1]-goals[0])):.3f}m "
        f"one_way={one_way_s:.2f}s cycles={a.cycles} steps={steps} speed={a.speed_mps:.3f}m/s",
        flush=True,
    )

    goal_log: list[np.ndarray] = []
    tool_log: list[np.ndarray] = []
    error_log: list[float] = []
    height_log: list[float] = []
    head_down_log: list[float] = []
    cylinder_log: list[np.ndarray] = []
    phase_log: list[str] = []
    for step in range(steps):
        started = time.perf_counter()
        elapsed = min(step * float(env.step_dt), total_s)
        cycle_t = elapsed % (2.0 * one_way_s)
        outward = cycle_t <= one_way_s
        stroke_t = cycle_t if outward else cycle_t - one_way_s
        profile_t = stroke_t if outward else one_way_s - stroke_t
        distance, _speed, _accel, _ = cruise_profile(
            torch.tensor([profile_t], device=env.device),
            length_m=arc_length,
            cruise_speed_mps=float(a.speed_mps),
            accel_time_s=float(a.accel_time_s),
            decel_time_s=float(a.accel_time_s),
        )
        goal_np, _seed_at_goal, _teacher_wz = interpolate(float(distance.item()))
        direction = 1.0 if outward else -1.0
        seed_distance = float(distance.item()) + direction * abs(float(_speed.item())) * float(
            a.seed_lead_time_s
        )
        _seed_goal, seed_np, _seed_wz = interpolate(seed_distance)
        goal = torch.as_tensor(goal_np, device=env.device).reshape(1, 3)
        seed = torch.as_tensor(seed_np, device=env.device).reshape(1, 4)
        cylinder_goal = cascade.cylinder_goal_seeded(
            goal, seed, correction_gain=float(a.ee_correction_gain)
        )
        cylinder_goal[:, 3] = seed[:, 3]
        action, _sv = cylinder.action(cylinder_goal)
        env._cylinder_goal_x[:] = cylinder_goal
        constrain_truck_xy(truck_anchor_xy)
        hold_wheels(wheel_target)
        env.step(action)

        tool = env._compute_ee_pos_local()[0]
        x, _xd = env._refresh_cylinder_state()
        err = float(torch.linalg.norm(tool - goal[0]).item())
        goal_log.append(goal_np.copy())
        tool_log.append(tool.detach().cpu().numpy().copy())
        error_log.append(err)
        height_log.append(float(env._compute_cutter_world_z()[0].item()))
        head_down_log.append(mower_down_alignment())
        cylinder_log.append(x[0].detach().cpu().numpy().copy())
        phase_log.append("A->B" if outward else "B->A")
        if step % 60 == 0:
            show_markers(markers, dense_path, goal[0])
        if step % 600 == 0 or step == steps - 1:
            drift_vector = (
                env.robot.data.body_link_state_w[0, env._truck_base_body_id(), :3] - truck_start
            )
            drift = float(torch.linalg.norm(drift_vector[:2]).item())
            print(
                f"[GROUND-ARC] step={step:05d}/{steps-1} phase={phase_log[-1]} "
                f"s={float(distance.item()):.3f}m error={err:.4f}m "
                f"world_z={height_log[-1]:+.3f}m truck_xy_drift={drift:.4f}m "
                f"truck_dz={float(drift_vector[2].item()):+.4f}m "
                f"head_down={head_down_log[-1]:.3f}",
                flush=True,
            )
        if bool(a.realtime):
            time.sleep(max(float(env.step_dt) - (time.perf_counter() - started), 0.0))

    G, E = np.asarray(goal_log), np.asarray(tool_log)
    error = np.asarray(error_log)
    height = np.asarray(height_log)
    head_down = np.asarray(head_down_log)
    cylinders = np.asarray(cylinder_log)
    warmup = min(120, max(len(error) // 20, 1))
    scored = error[warmup:]
    goal_path = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
    tool_path = float(np.linalg.norm(np.diff(E, axis=0), axis=1).sum())
    smooth_window = 15
    kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    filtered_tool = np.column_stack(
        [np.convolve(E[:, axis], kernel, mode="valid") for axis in range(3)]
    )
    filtered_tool_path = float(np.linalg.norm(np.diff(filtered_tool, axis=0), axis=1).sum())
    truck_drift_vector = (
        env.robot.data.body_link_state_w[0, env._truck_base_body_id(), :3] - truck_start
    )
    truck_drift = float(torch.linalg.norm(truck_drift_vector[:2]).item())
    mean = float(scored.mean())
    p95 = float(np.percentile(scored, 95))
    maximum = float(scored.max())
    final = float(error[-1])
    ground_fraction = float(np.mean(np.abs(height) <= float(a.ground_tolerance_m)))
    head_down_min = float(head_down.min())
    head_down_mean = float(head_down.mean())
    endpoint_a_error = float(np.linalg.norm(E[-1] - goals[0]))
    cylinder_travel = np.ptp(cylinders, axis=0)
    expected_path = 2.0 * arc_length * int(a.cycles)
    follow_ratio = filtered_tool_path / max(goal_path, 1e-9)
    nonvacuous = (
        goal_path >= 0.98 * expected_path
        and tool_path >= 0.75 * goal_path
        and follow_ratio <= 1.25
    )
    passed = bool(
        nonvacuous
        and mean < float(a.tracking_tolerance_m)
        and p95 < float(a.tracking_tolerance_m)
        and final < float(a.tracking_tolerance_m)
        and endpoint_a_error < float(a.tracking_tolerance_m)
        and ground_fraction >= 0.95
        and head_down_min >= float(a.head_down_min_alignment)
        and truck_drift < float(a.truck_drift_tolerance_m)
        and deployment_drift < 0.10
        and float(cylinder_travel[0]) > 0.20
    )
    result = dict(
        passed=passed,
        arc_length_m=arc_length,
        chord_m=float(np.linalg.norm(goals[-1] - goals[0])),
        cycles=int(a.cycles),
        one_way_s=one_way_s,
        goal_path_m=goal_path,
        tool_path_m=tool_path,
        filtered_tool_path_m=filtered_tool_path,
        follow_ratio=follow_ratio,
        error_mean_m=mean,
        error_p95_m=p95,
        error_max_m=maximum,
        error_final_m=final,
        endpoint_a_error_m=endpoint_a_error,
        ground_fraction=ground_fraction,
        cutter_world_z_min_m=float(height.min()),
        cutter_world_z_max_m=float(height.max()),
        head_down_min_alignment=head_down_min,
        head_down_mean_alignment=head_down_mean,
        head_down_max_tilt_deg=float(np.degrees(np.arccos(np.clip(head_down_min, -1.0, 1.0)))),
        truck_xy_drift_m=truck_drift,
        truck_drift_xyz_m=truck_drift_vector.detach().cpu().tolist(),
        deployment_truck_drift_m=deployment_drift,
        cylinder_travel_m=cylinder_travel.tolist(),
        active_id=str(cfg.id_cylinder_checkpoint),
        active_fd=str(cfg.fd_cylinder_checkpoint),
        reference_blend=float(cfg.cylinder_idfd_reference_blend),
        ee_correction_gain=float(a.ee_correction_gain),
        seed_lead_time_s=float(a.seed_lead_time_s),
    )
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[GROUND-ARC][RESULT] {'PASS' if passed else 'FAIL'} {json.dumps(result)}", flush=True)
    if not passed:
        raise SystemExit(2)


try:
    main()
finally:
    env.close()
    simulation_app.close()
