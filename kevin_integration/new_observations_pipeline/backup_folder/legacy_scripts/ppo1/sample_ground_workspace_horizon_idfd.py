#!/usr/bin/env python3
"""Sample safe reachable ground points through the active PPO1 Horizon-IDFD wrapper."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "on")


parser = argparse.ArgumentParser(description="Sample ground workspace for PPO1 Horizon-IDFD.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=200)
parser.add_argument("--max_attempts", type=int, default=1200)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--control_steps_per_target", type=int, default=8)
parser.add_argument("--tool_contact_mode", type=str, default="geometry_aabb", choices=["geometry_aabb", "manual_offset", "ee_origin"])
parser.add_argument("--tool_tip_local_offset_x", type=float, default=0.0)
parser.add_argument("--tool_tip_local_offset_y", type=float, default=0.0)
parser.add_argument("--tool_tip_local_offset_z", type=float, default=0.0)
parser.add_argument("--tool_ground_clearance", type=float, default=0.03)
parser.add_argument("--min_arm_ground_clearance", type=float, default=0.02)
parser.add_argument("--ground_z", type=float, default=0.0)
parser.add_argument("--height_tolerance", type=float, default=0.12)
parser.add_argument("--trajectory_scale", type=float, default=0.8)
parser.add_argument("--visualize_target_trajectory", type=str2bool, default=False)
parser.add_argument("--save_path", type=str, default="outputs/ppo1_horizon_idfd/ground_workspace.json")
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--ee_action_dim", type=int, default=3)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""):
    args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def robust_bounds(values: np.ndarray) -> tuple[float, float]:
    p5, p95 = np.percentile(values, [5, 95])
    margin = 0.10 * max(float(p95 - p5), 1.0e-6)
    return float(p5 + margin), float(p95 - margin)


def farthest_pair(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if len(points_xy) < 2:
        p = points_xy[0] if len(points_xy) else np.zeros(2, dtype=np.float64)
        return p, p, 0.0
    max_points = 1000
    pts = points_xy
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    diff = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
    return pts[i], pts[j], float(math.sqrt(float(dist2[i, j])))


def ensure_ground_z_in_workspace(env: HorizonIDFDEnv, target_world_z: float) -> None:
    root_xy = env._uw.robot.data.root_pos_w[0, :2].detach()
    probe_w = torch.tensor([[root_xy[0].item(), root_xy[1].item(), target_world_z]], device=env.device)
    target_base_z = float(env._world_to_base(probe_w)[0, 2].detach().cpu().item())
    wz = list(getattr(env.cfg, "horizon_workspace_z_range", (-0.4, 3.0)))
    if target_base_z < float(wz[0]) + 0.05:
        wz[0] = target_base_z - 0.20
        env.cfg.horizon_workspace_z_range = tuple(wz)
        print(f"[GROUND_WS] widened workspace z min to {wz[0]:.3f} for fixed world ground height", flush=True)


cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.ee_action_mode = "task_space_delta"
cfg.ee_action_dim = int(args_cli.ee_action_dim)
cfg.lowlevel_actuator_backend = str(args_cli.backend)
cfg.ppo_stage = "local_reach_stage0"
cfg.stage0_level = 0
cfg.tool_contact_mode = str(args_cli.tool_contact_mode)
cfg.tool_tip_local_offset_x = float(args_cli.tool_tip_local_offset_x)
cfg.tool_tip_local_offset_y = float(args_cli.tool_tip_local_offset_y)
cfg.tool_tip_local_offset_z = float(args_cli.tool_tip_local_offset_z)
cfg.tool_ground_clearance = float(args_cli.tool_ground_clearance)
cfg.min_arm_ground_clearance = float(args_cli.min_arm_ground_clearance)
cfg.ground_z = float(args_cli.ground_z)
cfg.trajectory_scale = float(args_cli.trajectory_scale)
cfg.visualize_target_trajectory = bool(args_cli.visualize_target_trajectory)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = True
cfg.horizon_reset_settle_steps = 0
cfg.device = args_cli.device
cfg.sim.device = args_cli.device
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = HorizonIDFDEnv(
    base_cfg=cfg,
    horizon_steps=int(args_cli.horizon_steps),
    action_scale_m=0.01,
    fd_gains=[2.0, 3.0, 3.0, 3.0],
    device=args_cli.device,
)

try:
    env.reset()
    frame = env.discover_controlled_frame()
    target_world_z = float(args_cli.ground_z) + float(args_cli.tool_ground_clearance)
    ensure_ground_z_in_workspace(env, target_world_z)
    print("[GROUND_WS] controlled frame discovery:", flush=True)
    for key, value in frame.items():
        print(f"  {key}: {value}", flush=True)
    if not bool(frame["can_represent_true_ground_xy_path"]):
        print("[GROUND_WS] LIMITATION: current interface cannot command true world ground XY paths.", flush=True)

    wx = getattr(env.cfg, "horizon_workspace_x_range", (-0.5, 7.0))
    wy = getattr(env.cfg, "horizon_workspace_y_range", (-3.6, 3.0))
    base_corners = torch.tensor(
        [[wx[0], wy[0], 0.0], [wx[0], wy[1], 0.0], [wx[1], wy[0], 0.0], [wx[1], wy[1], 0.0]],
        dtype=torch.float32,
        device=env.device,
    )
    world_corners = env._base_to_world(base_corners).detach().cpu().numpy()
    x_lo, x_hi = float(world_corners[:, 0].min()), float(world_corners[:, 0].max())
    y_lo, y_hi = float(world_corners[:, 1].min()), float(world_corners[:, 1].max())

    accepted: list[list[float]] = []
    rejected = 0
    rng = np.random.default_rng(42)
    attempts = 0
    while len(accepted) < int(args_cli.num_samples) and attempts < int(args_cli.max_attempts):
        attempts += 1
        target_w = torch.tensor(
            [[rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), target_world_z]],
            dtype=torch.float32,
            device=env.device,
        )
        env.set_ground_target_world(target_w)
        done_any = False
        for _ in range(int(args_cli.control_steps_per_target)):
            action = env.compute_ground_tracking_action(gain=1.0)
            _, _, terminated, truncated, _ = env.step(action)
            done_any = bool((terminated | truncated).any().detach().cpu().item())
            if done_any:
                env.reset()
                ensure_ground_z_in_workspace(env, target_world_z)
                break
        if done_any:
            rejected += 1
            continue

        target_now = env.current_ground_target_world()
        metrics = env.ground_safety_metrics(target_now)
        tool_world = env.get_tool_bottom_world_pos()[0].detach().cpu().numpy()
        safe = True
        safe &= bool(metrics["target_tool_bottom_clearance"][0].detach().cpu().item() >= float(args_cli.tool_ground_clearance) - 0.005)
        safe &= bool(metrics["tool_bottom_clearance"][0].detach().cpu().item() >= -0.01)
        safe &= bool(abs(float(tool_world[2]) - target_world_z) <= float(args_cli.height_tolerance))
        safe &= bool(metrics["arm_geometry_clearance"][0].detach().cpu().item() >= -0.02)
        safe &= bool(metrics["max_abs_body_world_position"][0].detach().cpu().item() < 100.0)
        safe &= bool(metrics["nan_inf_count"][0].detach().cpu().item() == 0)
        safe &= bool(metrics["cylinder_limit_fraction"][0].detach().cpu().item() == 0.0)
        safe &= bool(metrics["root_motion_norm"][0].detach().cpu().item() < 0.05)
        if safe:
            accepted.append([float(tool_world[0]), float(tool_world[1]), float(tool_world[2])])
            if len(accepted) % 25 == 0:
                print(f"[GROUND_WS] accepted={len(accepted)} rejected={rejected} attempts={attempts}", flush=True)
        else:
            rejected += 1

    accepted_np = np.asarray(accepted, dtype=np.float64)
    if len(accepted_np) == 0:
        tool = env.get_tool_bottom_world_pos()[0].detach().cpu().numpy()
        accepted_np = np.asarray([[tool[0], tool[1], max(tool[2], target_world_z)]], dtype=np.float64)
        print("[GROUND_WS] WARNING: no safe ground samples accepted; writing reduced fallback around current EE.", flush=True)

    sx0, sx1 = robust_bounds(accepted_np[:, 0])
    sy0, sy1 = robust_bounds(accepted_np[:, 1])
    if sx1 <= sx0:
        sx0 -= 0.05
        sx1 += 0.05
    if sy1 <= sy0:
        sy0 -= 0.05
        sy1 += 0.05
    safe_points = accepted_np[
        (accepted_np[:, 0] >= sx0) & (accepted_np[:, 0] <= sx1)
        & (accepted_np[:, 1] >= sy0) & (accepted_np[:, 1] <= sy1)
    ]
    if len(safe_points) == 0:
        safe_points = accepted_np
    p0_xy, p1_xy, line_len = farthest_pair(safe_points[:, :2])
    mid_xy = 0.5 * (p0_xy + p1_xy)
    p0_xy = mid_xy + 0.9 * (p0_xy - mid_xy)
    p1_xy = mid_xy + 0.9 * (p1_xy - mid_xy)
    center = [float(np.median(safe_points[:, 0])), float(np.median(safe_points[:, 1])), target_world_z]
    radius = max(0.02, 0.8 * min(center[0] - sx0, sx1 - center[0], center[1] - sy0, sy1 - center[1]))
    amp = max(0.01, 0.7 * 0.5 * (sy1 - sy0))
    acceptance_rate = float(len(accepted)) / max(float(len(accepted) + rejected), 1.0)
    workspace = {
        "ground_z": float(args_cli.ground_z),
        "tool_ground_clearance": float(args_cli.tool_ground_clearance),
        "tool_contact_mode": str(args_cli.tool_contact_mode),
        "target_world_z": float(target_world_z),
        "controlled_frame": frame["controlled_frame"],
        "controlled_axes": frame["controlled_axes"],
        "can_represent_true_ground_xy_path": frame["can_represent_true_ground_xy_path"],
        "safe_x_min": float(sx0),
        "safe_x_max": float(sx1),
        "safe_y_min": float(sy0),
        "safe_y_max": float(sy1),
        "safe_ground_points_count": int(len(accepted)),
        "rejected_points_count": int(rejected),
        "acceptance_rate": acceptance_rate,
        "largest_line_endpoints": [
            [float(p0_xy[0]), float(p0_xy[1]), target_world_z],
            [float(p1_xy[0]), float(p1_xy[1]), target_world_z],
        ],
        "largest_line_length": float(0.9 * line_len),
        "semicircle_center": center,
        "semicircle_radius": float(radius),
        "mower_sweep_bounds": {
            "x_min": float(sx0),
            "x_max": float(sx1),
            "y_min": float(sy0),
            "y_max": float(sy1),
        },
        "mower_sweep_amplitude": float(amp),
        "mower_sweep_length": float(sx1 - sx0),
        "accepted_ground_points": accepted[:1000],
    }
    save_path = Path(args_cli.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(workspace, indent=2) + "\n", encoding="utf-8")
    print(f"[GROUND_WS] saved {save_path}", flush=True)
    print(f"[GROUND_WS] accepted={len(accepted)} rejected={rejected} acceptance_rate={acceptance_rate:.3f}", flush=True)
    print(f"[GROUND_WS] safe_x=[{sx0:.3f}, {sx1:.3f}] safe_y=[{sy0:.3f}, {sy1:.3f}]", flush=True)
    print(f"[GROUND_WS] semicircle_radius={radius:.3f} line_length={0.9 * line_len:.3f} mower_amp={amp:.3f}", flush=True)
finally:
    env.close()
    simulation_app.close()
