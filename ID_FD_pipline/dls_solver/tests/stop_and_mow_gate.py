#!/usr/bin/env python3
"""Stop-and-mow controller gate: park, move the arm, advance, and repeat.

The truck is never allowed to contribute to the scored cutter trajectory.  Each
work phase is measured in the live ``base_link`` frame with the wheel parking
brake engaged.  Between work phases the arm holds its endpoint while the truck
drives straight to the next station; relocation has a separate geometry gate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--shape", choices=("arc", "curve", "line", "semicircle", "sweep"), default="sweep")
parser.add_argument("--stations", type=int, default=2)
parser.add_argument(
    "--arm_only", action=argparse.BooleanOptionalAction, default=True,
    help="Disable truck and run one long smooth mowing cell (default: enabled)",
)
parser.add_argument("--long_path_m", type=float, default=7.0, help="Minimum accumulated arm-only cutter travel")
parser.add_argument("--work_speed_mps", type=float, default=0.045)
parser.add_argument("--arc_length_m", type=float, default=1.51)
parser.add_argument("--line_length_m", type=float, default=None, help="Override auto-selected maximum unique span")
parser.add_argument("--line_cycles", type=int, default=1)
parser.add_argument("--curve_forward_bulge_m", type=float, default=0.10)
parser.add_argument("--truck_advance_m", type=float, default=0.50)
parser.add_argument("--max_advance_steps", type=int, default=3600)
parser.add_argument("--truck_speed_kmh", type=float, default=4.5)
parser.add_argument("--tolerance_m", type=float, default=0.05)
parser.add_argument("--map", default=str(ROOT / "ID_FD_pipline/models/ground_band_map.json"))
parser.add_argument("--realtime", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 1:
    raise ValueError("Stop-and-mow requires --num_envs 1")
if args.arm_only:
    args.stations = 1
    if args.shape not in ("line", "semicircle", "sweep"):
        raise ValueError("--arm_only supports line, semicircle, and sweep")
if args.line_length_m is None:
    args.line_length_m = 1.00 if args.arm_only and args.shape == "line" else 0.80 if args.arm_only else 0.35
if args.stations < 2 and not args.arm_only:
    raise ValueError("Use at least two work stations so relocation is exercised")

random.seed(args.seed)
np.random.seed(args.seed)
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import SPHERE_MARKER_CFG  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import ID_FD_pipline.tasks  # noqa: E402,F401
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController  # noqa: E402
from ID_FD_pipline.dls_solver.ee_cascade import EECascadeController  # noqa: E402
from ID_FD_pipline.dls_solver.ground_paths import GroundPath, GroundPathPlanner  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

TASK = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
cfg.seed = int(args.seed)
cfg.enable_scripted_vehicle_motion = False
cfg.vehicle_speed_min_kmh = float(args.truck_speed_kmh)
cfg.vehicle_speed_max_kmh = float(args.truck_speed_kmh)
cfg.vehicle_wheel_speed_gain = 1.0
cfg.episode_length_s = 100000.0
cfg.ee_success_hold_steps = 100000
cfg.low_level_servo_mode = "none"
env = gym.make(TASK, cfg=cfg).unwrapped

planner = GroundPathPlanner(args.map)
arc = planner.arc()
work_arc = arc
reset_jacobian = torch.tensor(
    json.loads((ROOT / "ID_FD_pipline/models/ee_jacobian_truck3x4.json").read_text())["jacobian"],
    dtype=torch.float32,
    device=env.device,
)
ground_jacobian_payload = json.loads(
    (ROOT / "ID_FD_pipline/models/ee_jacobian_ground_truck3x4.json").read_text()
)
ground_jacobian = torch.tensor(
    ground_jacobian_payload["jacobian"],
    dtype=torch.float32,
    device=env.device,
)
if args.arm_only:
    reduced_sv = torch.linalg.svdvals(ground_jacobian[:, :3].detach().cpu())
    if int(torch.linalg.matrix_rank(ground_jacobian[:, :3].detach().cpu())) != 3:
        raise RuntimeError("Cylinders 1-3 are not full-rank with cylinder 4 reserved for head pitch")
    ground_jacobian[:, 3] = 0.0
    print(
        f"[ARM-ONLY] cylinder4 reserved for head-down; position J123 "
        f"cond={float(reduced_sv[0] / reduced_sv[-1]):.2f}", flush=True
    )
# One small CPU decomposition, cached for the entire run.
ground_pinv = torch.linalg.pinv(ground_jacobian.detach().cpu()).numpy()
base_id = env._truck_base_body_id()

MIN_WORK_PATH_M = 1.50
MIN_SWEEP_PATH_PER_STATION_M = 0.65
MIN_GROUND_FRACTION = 0.90
MAX_PARKED_TRUCK_DRIFT_M = 0.10
MAX_ADVANCE_LATERAL_M = 0.25
MAX_ADVANCE_HEADING_DEG = 10.0


def set_parking_brake(enabled: bool) -> None:
    wheel_ids = env._wheel_joint_ids
    parked_stiffness = 6.0e8 if args.shape == "arc" else 2.0e9
    parked_damping = 1.0e6 if args.shape == "arc" else 1.0e7
    env.robot.write_joint_stiffness_to_sim(parked_stiffness if enabled else 0.0, joint_ids=wheel_ids)
    env.robot.write_joint_damping_to_sim(parked_damping if enabled else 1.0e6, joint_ids=wheel_ids)
    if enabled:
        wheel_pos = env.robot.data.joint_pos[:, wheel_ids].detach().clone()
        env.robot.set_joint_position_target(wheel_pos, joint_ids=wheel_ids)


def mower_down_axis() -> torch.Tensor:
    """Messerkopf authored local +Z expressed in world coordinates."""
    q = env.robot.data.body_link_state_w[0, env._ee_body_id, 3:7]
    q = q / q.norm().clamp(min=1.0e-8)
    return quat_apply(q.reshape(1, 4), torch.tensor([[0.0, 0.0, 1.0]], device=env.device))[0]


def sample_direct_target(path: GroundPath, distance_m: float) -> np.ndarray:
    distance_m = float(np.clip(distance_m, 0.0, path.length_m))
    upper = int(np.searchsorted(path.cumulative_m, distance_m, side="right"))
    upper = min(max(upper, 1), len(path.cumulative_m) - 1)
    lower = upper - 1
    span = max(float(path.cumulative_m[upper] - path.cumulative_m[lower]), 1.0e-9)
    alpha = (distance_m - float(path.cumulative_m[lower])) / span
    return (1.0 - alpha) * path.direct_targets[lower] + alpha * path.direct_targets[upper]


def truck_to_world(points: torch.Tensor) -> torch.Tensor:
    base = env.robot.data.body_link_state_w[0, base_id]
    q = base[3:7] / base[3:7].norm().clamp(min=1.0e-8)
    return base[:3] + quat_apply(q.reshape(1, 4).expand(len(points), -1), points)


def make_markers():
    if bool(args.headless):
        return None
    configs = []
    for path_name, radius, color in (
        ("/Visuals/StopAndMowPath", 0.035, (0.1, 0.9, 0.2)),
        ("/Visuals/StopAndMowGoal", 0.10, (1.0, 0.1, 0.1)),
        ("/Visuals/StopAndMowCutter", 0.08, (0.1, 0.4, 1.0)),
    ):
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.markers["sphere"].radius = radius
        marker_cfg.markers["sphere"].visual_material.diffuse_color = color
        configs.append(VisualizationMarkers(marker_cfg.replace(prim_path=path_name)))
    base = env.robot.data.body_link_state_w[0, base_id, :3]
    env.sim.set_camera_view(
        (base + torch.tensor([10.0, -12.0, 8.0], device=env.device)).cpu().tolist(),
        (base + torch.tensor([4.0, 0.0, 0.5], device=env.device)).cpu().tolist(),
    )
    return tuple(configs)


def show_markers(markers, local_path: np.ndarray, goal: torch.Tensor) -> None:
    if markers is None:
        return
    path_marker, goal_marker, cutter_marker = markers
    points = torch.as_tensor(local_path, device=env.device, dtype=torch.float32)
    path_marker.visualize(truck_to_world(points))
    goal_marker.visualize(truck_to_world(goal.reshape(1, 3)))
    cutter_marker.visualize(env._compute_cutter_pos_w())


def build_controller(jacobian: torch.Tensor) -> tuple[CylinderDLSController, EECascadeController]:
    controller = CylinderDLSController(
        env,
        CylinderDLSConfig(
            invert_learned_pipeline=False,
            error_gain=0.10,
            max_desired_delta_m=0.01,
            max_action=1.0,
            action_smoothing=0.35,
        ),
    )
    cascade = EECascadeController(
        env,
        cylinder_controller=controller,
        jacobian=jacobian,
        ee_feedback_gain=0.10 if args.shape in ("curve", "line", "semicircle", "sweep") else 0.12,
        max_ik_step_m=0.01,
        max_cylinder_lead_m=0.01,
    )
    cascade.reset()
    return controller, cascade


@torch.no_grad()
def hold_goal(
    controller: CylinderDLSController,
    cascade: EECascadeController,
    goal: torch.Tensor,
    seed: torch.Tensor | None,
    steps: int,
) -> None:
    for _ in range(int(steps)):
        cylinder_goal = (
            cascade.cylinder_goal_seeded(goal, seed, correction_gain=0.40)
            if seed is not None
            else cascade.cylinder_goal(goal)
        )
        controller.step(cylinder_goal)


def work_reference(station: int, step: int, anchor_goal: np.ndarray, anchor_seed: np.ndarray, line_direction: np.ndarray):
    dt = float(env.step_dt)
    speed = (
        float(args.work_speed_mps) if args.shape == "arc"
        else 0.02 if args.arm_only
        else 0.04
    )
    progress = float(step) * dt * speed
    if args.shape == "arc":
        length = min(float(args.arc_length_m), work_arc.length_m)
        distance = min(progress, length)
        goal, seed, _ = work_arc.sample(float(np.clip(distance, 0.0, work_arc.length_m)))
        return goal, seed, float(np.clip(progress, 0.0, length))
    if args.arm_only:
        length = float(args.line_length_m)
        if args.shape == "line":
            half = 0.5 * length
            phase = progress % (2.0 * length)
            segment = min(int(phase / max(half, 1.0e-9)), 3)
            raw = (phase - segment * half) / max(half, 1.0e-9)
            smooth = 0.5 * (1.0 - math.cos(math.pi * raw))
            endpoints = ((0.0, half), (half, 0.0), (0.0, -half), (-half, 0.0))
            start, end = endpoints[segment]
            along = start + (end - start) * smooth
            unit = (along + half) / max(length, 1.0e-9)
        else:
            phase = progress % (2.0 * length)
            stroke = phase if phase <= length else 2.0 * length - phase
            raw = stroke / max(length, 1.0e-9)
            unit = 0.5 * (1.0 - math.cos(math.pi * raw))
            along = length * unit
        forward = 0.0
        if args.shape == "semicircle":
            forward = 0.18 * math.sin(math.pi * unit)
        elif args.shape == "sweep":
            # Three complete forward/back lobes across one lateral stroke.
            forward = 0.06 * math.sin(6.0 * math.pi * unit)
        goal = anchor_goal + along * line_direction
        goal = goal + forward * np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        seed = anchor_seed + (goal - anchor_goal) @ ground_pinv.T
        seed[3] = anchor_seed[3]
        return goal.astype(np.float32), seed.astype(np.float32), progress
    if args.shape == "curve":
        lateral_reach = float(args.line_length_m)
        # A bowed mowing stroke in front of the parked truck, followed by the
        # exact reverse stroke.  This stays on the robust lateral linkage branch
        # while making the cutter (not the truck) produce the curved geometry.
        phase = progress % (2.0 * lateral_reach)
        curve_distance = phase if phase <= lateral_reach else 2.0 * lateral_reach - phase
        unit = curve_distance / max(lateral_reach, 1.0e-9)
        forward_dir = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        goal = (
            anchor_goal
            + curve_distance * line_direction
            + float(args.curve_forward_bulge_m) * math.sin(math.pi * unit) * forward_dir
        )
        seed = anchor_seed + (goal - anchor_goal) @ ground_pinv.T
        return goal.astype(np.float32), seed.astype(np.float32), progress
    length = float(args.line_length_m)
    phase = progress % (2.0 * length)
    along = phase if phase <= length else 2.0 * length - phase
    goal = anchor_goal + along * line_direction
    # First-order mapped seed is only feedforward; measured EE feedback closes IK.
    seed = anchor_seed + (goal - anchor_goal) @ ground_pinv.T
    return goal.astype(np.float32), seed.astype(np.float32), progress


def local_reference_points(station: int, anchor_goal: np.ndarray, line_direction: np.ndarray) -> np.ndarray:
    if args.shape == "arc":
        length = min(float(args.arc_length_m), work_arc.length_m)
        query = np.linspace(0.0, length, 100)
        return np.asarray([work_arc.sample(float(value))[0] for value in query])
    if args.arm_only:
        unit = np.linspace(0.0, 1.0, 180)
        along = float(args.line_length_m) * unit
        if args.shape == "line":
            along = float(args.line_length_m) * (unit - 0.5)
        points = anchor_goal[None, :] + along[:, None] * line_direction[None, :]
        if args.shape == "semicircle":
            points[:, 0] += 0.18 * np.sin(np.pi * unit)
        elif args.shape == "sweep":
            points[:, 0] += 0.06 * np.sin(6.0 * np.pi * unit)
        return points
    if args.shape == "curve":
        lateral_reach = float(args.line_length_m)
        unit = np.linspace(0.0, 1.0, 80)
        return (
            anchor_goal[None, :]
            + (lateral_reach * unit)[:, None] * line_direction[None, :]
            + (float(args.curve_forward_bulge_m) * np.sin(math.pi * unit))[:, None]
            * np.asarray([[1.0, 0.0, 0.0]])
        )
    along = np.linspace(0.0, float(args.line_length_m), 60)
    return anchor_goal[None, :] + along[:, None] * line_direction[None, :]


@torch.no_grad()
def run_work_station(
    station: int,
    controller: CylinderDLSController,
    cascade: EECascadeController,
    anchor_goal: np.ndarray,
    anchor_seed: np.ndarray,
    line_direction: np.ndarray,
    markers,
) -> tuple[dict[str, float | bool], torch.Tensor, torch.Tensor | None]:
    set_parking_brake(True)
    env.cfg.enable_scripted_vehicle_motion = False
    controller.reset(); cascade.reset()
    local_path = local_reference_points(station, anchor_goal, line_direction)
    if args.arm_only:
        one_stroke = local_reference_points(station, anchor_goal, line_direction)
        cycle_path = 2.0 * float(np.linalg.norm(np.diff(one_stroke, axis=0), axis=1).sum())
        cycles = int(math.ceil(float(args.long_path_m) / max(cycle_path, 1.0e-9)))
        work_length = 2.0 * float(args.line_length_m) * cycles
    elif args.shape == "arc":
        work_length = min(float(args.arc_length_m), work_arc.length_m)
    elif args.shape == "curve":
        lateral_reach = float(args.line_length_m)
        # This is the reference parameter length.  The measured Cartesian path
        # is longer because of the forward bow and is computed from the goals.
        work_length = 2.0 * lateral_reach
    else:
        work_length = 2.0 * float(args.line_length_m) * int(args.line_cycles)
    speed = (
        float(args.work_speed_mps) if args.shape == "arc"
        else 0.02 if args.arm_only
        else 0.04
    )
    steps = int(math.ceil(work_length / (speed * float(env.step_dt)))) + 1
    start_truck = env.robot.data.body_link_state_w[0, base_id, :3].detach().clone()
    goals: list[np.ndarray] = []
    ees: list[np.ndarray] = []
    errors: list[float] = []
    heights: list[float] = []
    head_down_alignments: list[float] = []
    cylinders: list[np.ndarray] = []
    last_goal = None
    last_seed = None
    for step in range(steps):
        started = time.perf_counter()
        goal_np, seed_np, path_progress = work_reference(station, step, anchor_goal, anchor_seed, line_direction)
        if args.shape in ("curve", "line", "semicircle", "sweep"):
            # Close on the physical cutter-bottom clearance, which changes with
            # head orientation even when the body-fixed Cartesian point tracks.
            clearance = float(env._compute_cutter_world_z()[0].item())
            goal_np = goal_np.copy()
            goal_np[2] += float(np.clip(0.05 - clearance, -0.12, 0.12))
        goal = torch.as_tensor(goal_np, device=env.device, dtype=torch.float32).reshape(1, 3)
        seed = torch.as_tensor(seed_np, device=env.device, dtype=torch.float32).reshape(1, 4)
        cylinder_goal = (
            cascade.cylinder_goal_seeded(goal, seed, correction_gain=0.40)
            if args.shape == "arc"
            else cascade.cylinder_goal(goal)
        )
        controller.step(cylinder_goal)
        ee = env._compute_ee_pos_local()[0]
        x, _ = env._refresh_cylinder_state()
        goals.append(goal_np.copy()); ees.append(ee.detach().cpu().numpy().copy())
        errors.append(float(torch.linalg.norm(goal[0] - ee).item()))
        heights.append(float(env._compute_cutter_world_z()[0].item()))
        head_down_axis = mower_down_axis()
        head_down_alignments.append(float(-head_down_axis[2].item()))
        cylinders.append(x[0].detach().cpu().numpy().copy())
        last_goal, last_seed = goal, seed if args.shape == "arc" else None
        if step % 120 == 0:
            show_markers(markers, local_path, goal[0])
        if step % 600 == 0:
            print(
                f"[STOP-MOW][WORK] station={station} step={step}/{steps-1} "
                f"arm_path={path_progress:.3f}m error={errors[-1]:.4f}m z={heights[-1]:+.3f}m",
                flush=True,
            )
        if bool(args.realtime):
            time.sleep(max(float(env.step_dt) - (time.perf_counter() - started), 0.0))
    G = np.asarray(goals); E = np.asarray(ees); Z = np.asarray(heights); C = np.asarray(cylinders)
    H = np.asarray(head_down_alignments, dtype=np.float64)
    goal_path = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
    ee_path = float(np.linalg.norm(np.diff(E, axis=0), axis=1).sum())
    steady = np.asarray(errors[200:], dtype=np.float64)
    mean = float(steady.mean()); p95 = float(np.percentile(steady, 95.0))
    final = float(np.mean(steady[-50:])); tail_std = float(np.std(steady[-200:]))
    ground_fraction = float(np.mean(np.abs(Z) <= 0.10))
    cylinder_travel = np.ptp(C, axis=0)
    head_down_min = float(H.min())
    head_down_final = float(H[-50:].mean())
    truck_drift = float(torch.linalg.norm(env.robot.data.body_link_state_w[0, base_id, :3] - start_truck).item())
    follow_ratio = ee_path / max(goal_path, 1.0e-9)
    forward_min = float(G[:, 0].min()); forward_max = float(G[:, 0].max())
    lateral_min = float(G[:, 1].min()); lateral_max = float(G[:, 1].max())
    curve_in_front = args.shape not in ("arc", "curve", "semicircle") or forward_min > 0.50
    sweep_closed = args.shape == "arc" or float(np.linalg.norm(G[-1] - G[0])) < 0.05
    required_station_path = (
        float(args.long_path_m) if args.arm_only
        else MIN_WORK_PATH_M if args.shape == "arc"
        else MIN_SWEEP_PATH_PER_STATION_M
    )
    passed = bool(
        goal_path >= required_station_path
        and follow_ratio >= 0.60
        and p95 < float(args.tolerance_m)
        and final < float(args.tolerance_m)
        and tail_std < 0.02
        and ground_fraction >= MIN_GROUND_FRACTION
        and float(cylinder_travel.max()) > 0.03
        and (not args.arm_only or (head_down_min >= 0.90 and head_down_final >= 0.95 and float(cylinder_travel[3]) < 0.01))
        and (not args.arm_only or float(cylinder_travel[2]) > 0.04)
        and (not args.arm_only or args.shape == "line" or float(cylinder_travel[0]) > 0.0075)
        and truck_drift < MAX_PARKED_TRUCK_DRIFT_M
        and curve_in_front
        and sweep_closed
    )
    print(
        f"[STOP-MOW][WORK-RESULT] station={station} {'PASS' if passed else 'FAIL'} "
        f"goal_path={goal_path:.3f}m ee_path={ee_path:.3f}m follow={follow_ratio:.2f} "
        f"mean={mean:.4f} p95={p95:.4f} final={final:.4f} tail_std={tail_std:.4f} "
        f"ground={ground_fraction:.3f} truck_drift={truck_drift:.3f}m "
        f"truck_frame_x=[{forward_min:.3f},{forward_max:.3f}]m "
        f"truck_frame_y=[{lateral_min:.3f},{lateral_max:.3f}]m "
        f"curve_in_front={curve_in_front} sweep_closed={sweep_closed} "
        f"head_down_min={head_down_min:.3f} head_down_final={head_down_final:.3f} "
        f"cylinder_travel={np.round(cylinder_travel,4).tolist()}m",
        flush=True,
    )
    return {"passed": passed, "mean": mean, "p95": p95, "goal_path": goal_path}, last_goal, last_seed


@torch.no_grad()
def advance_truck(
    station: int,
    controller: CylinderDLSController,
    cascade: EECascadeController,
    hold: torch.Tensor,
    seed: torch.Tensor | None,
) -> dict[str, float | bool]:
    set_parking_brake(False)
    env.cfg.enable_scripted_vehicle_motion = True
    env.cfg.vehicle_wheel_speed_gain = 1.0
    env.cfg.vehicle_steering_target_rad = 0.0
    env.cfg.vehicle_left_wheel_speed_scale = 1.0
    env.cfg.vehicle_right_wheel_speed_scale = 1.0
    env.robot.write_joint_stiffness_to_sim(1.0e7, joint_ids=env._steering_joint_ids)
    env.robot.write_joint_damping_to_sim(1.0e6, joint_ids=env._steering_joint_ids)
    travel_steering = torch.tensor([[-0.05, 0.05]], device=env.device)
    env._scripted_steering_target = travel_steering.clone()
    start = env.robot.data.body_link_state_w[0, base_id].detach().clone()
    q0 = start[3:7] / start[3:7].norm().clamp(min=1.0e-8)
    forward0 = quat_apply(q0.reshape(1, 4), torch.tensor([[1.0, 0.0, 0.0]], device=env.device))[0]
    lateral0 = quat_apply(q0.reshape(1, 4), torch.tensor([[0.0, 1.0, 0.0]], device=env.device))[0]
    progress = 0.0
    lateral = 0.0
    heading_deg = 0.0
    env.cfg.vehicle_steering_target_rad = 0.0
    steps = 0
    for steps in range(1, int(args.max_advance_steps) + 1):
        env._scripted_steering_target[:] = travel_steering
        cylinder_goal = (
            cascade.cylinder_goal_seeded(hold, seed, correction_gain=0.40)
            if seed is not None
            else cascade.cylinder_goal(hold)
        )
        controller.step(cylinder_goal)
        state = env.robot.data.body_link_state_w[0, base_id]
        delta = state[:3] - start[:3]
        progress = abs(float(torch.dot(delta, forward0).item()))
        lateral = float(torch.dot(delta, lateral0).item())
        q = state[3:7] / state[3:7].norm().clamp(min=1.0e-8)
        forward = quat_apply(q.reshape(1, 4), torch.tensor([[1.0, 0.0, 0.0]], device=env.device))[0]
        heading_deg = math.degrees(math.acos(float(torch.clamp(torch.dot(forward, forward0), -1.0, 1.0).item())))
        if steps % 300 == 0:
            print(
                f"[STOP-MOW][ADVANCE] station={station} step={steps} forward={progress:.3f}m "
                f"lateral={lateral:+.3f}m heading={heading_deg:.2f}deg",
                flush=True,
            )
        if progress >= float(args.truck_advance_m):
            break
        if bool(args.realtime):
            time.sleep(float(env.step_dt))
    env.cfg.enable_scripted_vehicle_motion = False
    env.cfg.vehicle_wheel_speed_gain = 1.0
    env.cfg.vehicle_steering_target_rad = 0.0
    env.cfg.vehicle_left_wheel_speed_scale = 1.0
    env.cfg.vehicle_right_wheel_speed_scale = 1.0
    env._scripted_steering_target.zero_()
    env.robot.set_joint_velocity_target(
        torch.zeros((1, len(env._wheel_joint_ids)), device=env.device),
        joint_ids=env._wheel_joint_ids,
    )
    set_parking_brake(True)
    brake_steps = 0
    for brake_steps in range(1, 901):
        cylinder_goal = (
            cascade.cylinder_goal_seeded(hold, seed, correction_gain=0.40)
            if seed is not None
            else cascade.cylinder_goal(hold)
        )
        controller.step(cylinder_goal)
        speed_mps = float(torch.linalg.norm(env.robot.data.body_link_state_w[0, base_id, 7:10]).item())
        if brake_steps >= 120 and speed_mps < 0.02:
            break
    final_state = env.robot.data.body_link_state_w[0, base_id]
    final_delta = final_state[:3] - start[:3]
    progress = abs(float(torch.dot(final_delta, forward0).item()))
    lateral = abs(float(torch.dot(final_delta, lateral0).item()))
    q_final = final_state[3:7] / final_state[3:7].norm().clamp(min=1.0e-8)
    forward_final = quat_apply(
        q_final.reshape(1, 4), torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
    )[0]
    heading_deg = math.degrees(
        math.acos(float(torch.clamp(torch.dot(forward_final, forward0), -1.0, 1.0).item()))
    )
    straight_ratio = lateral / max(progress, 1.0e-9)
    env._scripted_steering_target = None
    env.robot.write_joint_stiffness_to_sim(10.0, joint_ids=env._steering_joint_ids)
    env.robot.write_joint_damping_to_sim(1.0e9, joint_ids=env._steering_joint_ids)
    passed = bool(
        progress >= float(args.truck_advance_m)
        and lateral <= MAX_ADVANCE_LATERAL_M
        and straight_ratio <= 0.20
    )
    print(
        f"[STOP-MOW][ADVANCE-RESULT] station={station} {'PASS' if passed else 'FAIL'} "
        f"drive_steps={steps} brake_steps={brake_steps} forward={progress:.3f}m "
        f"lateral={lateral:.3f}m lateral_ratio={straight_ratio:.2f} "
        f"heading_reported={heading_deg:.2f}deg",
        flush=True,
    )
    return {"passed": passed, "forward": progress, "lateral": lateral, "heading_deg": heading_deg}


@torch.no_grad()
def home_arc_for_next_station(
    controller: CylinderDLSController,
    cascade: EECascadeController,
    target: torch.Tensor,
    goal: torch.Tensor,
    seed: torch.Tensor,
) -> None:
    """Physically home the rams while parked; this phase is never scored."""
    driver = DirectCylinderDriver(env)
    for _ in range(40):
        driver.step(target)
    env._cylinder_pos_target[:] = env.robot.data.joint_pos[:, env._cylinder_joint_ids]
    env._prev_valve_cmd.zero_(); env._cylinder_id.reset(); env._cylinder_fd.reset()
    controller.reset(); cascade.reset()
    hold_goal(controller, cascade, goal, seed, 300)
    error = float(torch.linalg.norm(goal[0] - env._compute_ee_pos_local()[0]).item())
    print(
        f"[STOP-MOW][HOME] stepped_physx=True truck_braked=True initial_goal_error={error:.4f}m",
        flush=True,
    )


@torch.no_grad()
def stow_arc_for_travel(
    controller: CylinderDLSController,
    cascade: EECascadeController,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move to the measured compact ground pose that gives stable truck traction."""
    safe_x = torch.as_tensor(
        ground_jacobian_payload["cylinder_x"], device=env.device, dtype=torch.float32
    ).reshape(1, 4)
    driver = DirectCylinderDriver(env)
    for _ in range(40):
        driver.step(safe_x)
    env._cylinder_pos_target[:] = env.robot.data.joint_pos[:, env._cylinder_joint_ids]
    env._prev_valve_cmd.zero_(); env._cylinder_id.reset(); env._cylinder_fd.reset()
    controller.reset(); cascade.reset()
    goal = env._compute_ee_pos_local().detach().clone()
    seed = env._refresh_cylinder_state()[0].detach().clone()
    hold_goal(controller, cascade, goal, seed, 180)
    print(
        f"[STOP-MOW][STOW] travel_pose=True cutter_z={float(env._compute_cutter_world_z()[0].item()):+.3f}m",
        flush=True,
    )
    return goal, seed


@torch.no_grad()
def reset_learned_actuator_for_station(
    controller: CylinderDLSController,
    cascade: EECascadeController,
) -> None:
    """Clear travel-phase hydraulic history without changing the physical pose."""
    env._cylinder_pos_target[:] = env.robot.data.joint_pos[:, env._cylinder_joint_ids]
    env._prev_valve_cmd.zero_(); env._cylinder_id.reset(); env._cylinder_fd.reset()
    controller.reset(); cascade.reset()
    print("[STOP-MOW][STATION-RESET] learned actuator history cleared; pose unchanged", flush=True)


@torch.no_grad()
def stow_local_for_travel(
    controller: CylinderDLSController,
    cascade: EECascadeController,
    anchor_goal: np.ndarray,
    anchor_seed: np.ndarray,
) -> tuple[torch.Tensor, None]:
    """Return every ram, including the EE-null ram, to the stable travel pose."""
    set_parking_brake(True)
    env.cfg.enable_scripted_vehicle_motion = False
    target = torch.as_tensor(anchor_seed, device=env.device, dtype=torch.float32).reshape(1, 4)
    controller.reset(); cascade.reset()
    for _ in range(600):
        controller.step(target)
    measured, _ = env._refresh_cylinder_state()
    cylinder_error = float(torch.linalg.norm(target[0] - measured[0]).item())
    goal = torch.as_tensor(anchor_goal, device=env.device, dtype=torch.float32).reshape(1, 3)
    goal[:, 2] += 0.20
    controller.reset(); cascade.reset()
    hold_goal(controller, cascade, goal, None, 600)
    print(
        f"[STOP-MOW][STOW] learned_cylinder_dls=True cylinder_error={cylinder_error:.4f}m "
        f"cutter_z={float(env._compute_cutter_world_z()[0].item()):+.3f}m",
        flush=True,
    )
    return goal, None


@torch.no_grad()
def prepare_local_work(
    controller: CylinderDLSController,
    cascade: EECascadeController,
    anchor_goal: np.ndarray,
) -> None:
    """Lower the cutter to the station anchor before starting scored work."""
    set_parking_brake(True)
    env.cfg.enable_scripted_vehicle_motion = False
    goal = torch.as_tensor(anchor_goal, device=env.device, dtype=torch.float32).reshape(1, 3)
    controller.reset(); cascade.reset()
    hold_goal(controller, cascade, goal, None, 600)
    print(
        f"[STOP-MOW][DEPLOY] truck_braked=True cutter_z="
        f"{float(env._compute_cutter_world_z()[0].item()):+.3f}m",
        flush=True,
    )


try:
    env.cfg.cylinder_fd_residual_output_gain = [1.0, 1.0, 1.0, 1.0]
    run_seed = int(args.seed) + (11 if args.shape == "arc" else 23)
    random.seed(run_seed); np.random.seed(run_seed); torch.manual_seed(run_seed)
    env.reset(seed=run_seed)
    set_parking_brake(True)
    env.cfg.enable_scripted_vehicle_motion = False
    zero = torch.zeros((1, 4), device=env.device)
    for _ in range(40):
        env.step(zero)

    measured_x, _ = env._refresh_cylinder_state()
    if torch.linalg.norm(measured_x[0] - torch.as_tensor(arc.cylinder_seeds[-1], device=env.device)) < torch.linalg.norm(
        measured_x[0] - torch.as_tensor(arc.cylinder_seeds[0], device=env.device)
    ):
        work_arc = arc.reversed()
        print("[STOP-MOW] selected reversed mapped arc (closest reset branch)", flush=True)

    if args.shape == "arc":
        anchor_distance = 0.0
        jacobian = reset_jacobian
        anchor_goal, anchor_seed, _ = work_arc.sample(anchor_distance)
        anchor_target = sample_direct_target(work_arc, anchor_distance)
        controller, cascade = build_controller(jacobian)
        driver = DirectCylinderDriver(env)
        target = torch.as_tensor(anchor_target, device=env.device, dtype=torch.float32).reshape(1, 4)
        for _ in range(40):
            driver.step(target)
        env._cylinder_pos_target[:] = env.robot.data.joint_pos[:, env._cylinder_joint_ids]
        env._prev_valve_cmd.zero_(); env._cylinder_id.reset(); env._cylinder_fd.reset()
        controller.reset(); cascade.reset()
        line_direction = np.zeros(3, dtype=np.float32)
    else:
        jacobian = ground_jacobian
        controller, cascade = build_controller(jacobian)
        anchor_tensor = env._compute_ee_pos_local().detach().clone()
        drop = max(float(env._compute_cutter_world_z()[0].item()) - 0.05, 0.0)
        anchor_tensor[:, 2] -= drop
        for _ in range(250):
            controller.step(cascade.cylinder_goal(anchor_tensor))
        controller.reset(); cascade.reset()
        anchor_goal = anchor_tensor[0].detach().cpu().numpy().copy()
        anchor_seed = env._refresh_cylinder_state()[0][0].detach().cpu().numpy().copy()
        anchor_target = np.zeros(4, dtype=np.float32)
        line_direction = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    markers = make_markers()
    work_results = []
    advance_results = []
    hold = torch.as_tensor(anchor_goal, device=env.device, dtype=torch.float32).reshape(1, 3)
    seed = torch.as_tensor(anchor_seed, device=env.device, dtype=torch.float32).reshape(1, 4) if args.shape == "arc" else None
    for station in range(int(args.stations)):
        if station > 0 and args.shape == "curve":
            prepare_local_work(controller, cascade, anchor_goal)
        result, hold, seed = run_work_station(
            station, controller, cascade, anchor_goal, anchor_seed, line_direction, markers
        )
        work_results.append(result)
        if station + 1 < int(args.stations):
            if args.shape == "arc":
                hold, seed = stow_arc_for_travel(controller, cascade)
            elif args.shape == "curve":
                hold, seed = stow_local_for_travel(
                    controller, cascade, anchor_goal, anchor_seed
                )
            advance_results.append(advance_truck(station, controller, cascade, hold, seed))
            if args.shape == "arc":
                anchor_goal_tensor = torch.as_tensor(
                    anchor_goal, device=env.device, dtype=torch.float32
                ).reshape(1, 3)
                anchor_seed_tensor = torch.as_tensor(
                    anchor_seed, device=env.device, dtype=torch.float32
                ).reshape(1, 4)
                home_arc_for_next_station(
                    controller,
                    cascade,
                    torch.as_tensor(anchor_target, device=env.device, dtype=torch.float32).reshape(1, 4),
                    anchor_goal_tensor,
                    anchor_seed_tensor,
                )
                hold, seed = anchor_goal_tensor, anchor_seed_tensor
            reset_learned_actuator_for_station(controller, cascade)
    total_work_path = float(sum(float(item["goal_path"]) for item in work_results))
    aggregate_minimum = float(args.long_path_m) if args.arm_only else MIN_WORK_PATH_M
    aggregate_work_ok = total_work_path >= aggregate_minimum
    passed = all(bool(item["passed"]) for item in work_results + advance_results) and aggregate_work_ok
    print(
        f"[STOP-MOW][TOTAL-WORK] path={total_work_path:.3f}m "
        f"minimum={aggregate_minimum:.3f}m -> {'PASS' if aggregate_work_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"[STOP-MOW][GATE] {'PASS' if passed else 'FAIL'} shape={args.shape} "
        f"stations={args.stations} advances={len(advance_results)}",
        flush=True,
    )
    if not passed:
        raise SystemExit(2)
finally:
    env.close()
    simulation_app.close()
