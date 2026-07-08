#!/usr/bin/env python3
"""Debug visible ground trajectories through the PPO1 Horizon-IDFD wrapper."""
from __future__ import annotations

import argparse
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


parser = argparse.ArgumentParser(description="Debug PPO1 Horizon-IDFD ground trajectory tracking.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--num_steps", type=int, default=240)
parser.add_argument("--trajectory_mode", type=str, default="ground_semicircle",
                    choices=["ground_semicircle", "ground_straight_line", "ground_mower_sweep"])
parser.add_argument("--trajectory_plane", type=str, default="ground_xy", choices=["ground_xy", "base_yz"])
parser.add_argument("--ground_workspace_json", type=str, default="outputs/ppo1_horizon_idfd/ground_workspace.json")
parser.add_argument("--tool_contact_mode", type=str, default="geometry_aabb", choices=["geometry_aabb", "manual_offset", "ee_origin"])
parser.add_argument("--tool_tip_local_offset_x", type=float, default=0.0)
parser.add_argument("--tool_tip_local_offset_y", type=float, default=0.0)
parser.add_argument("--tool_tip_local_offset_z", type=float, default=0.0)
parser.add_argument("--tool_ground_clearance", type=float, default=0.03)
parser.add_argument("--min_arm_ground_clearance", type=float, default=0.02)
parser.add_argument("--debug_draw_tool_aabb", type=str2bool, default=False)
parser.add_argument("--debug_draw_tool_contact_point", type=str2bool, default=False)
parser.add_argument("--ground_z", type=float, default=0.0)
parser.add_argument("--trajectory_scale", type=float, default=0.8)
parser.add_argument("--visualize_target_trajectory", type=str2bool, default=True)
parser.add_argument("--visualize_actual_ee_trail", type=str2bool, default=True)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--ee_action_dim", type=int, default=3)
parser.add_argument("--require_path_completion", type=str2bool, default=True)
parser.add_argument("--fast_exit", action="store_true")
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""):
    args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def ensure_ground_z_in_workspace(env: HorizonIDFDEnv, target_world_z: float) -> None:
    root_xy = env._uw.robot.data.root_pos_w[0, :2].detach()
    probe_w = torch.tensor([[root_xy[0].item(), root_xy[1].item(), target_world_z]], device=env.device)
    target_base_z = float(env._world_to_base(probe_w)[0, 2].detach().cpu().item())
    wz = list(getattr(env.cfg, "horizon_workspace_z_range", (-0.4, 3.0)))
    if target_base_z < float(wz[0]) + 0.05:
        wz[0] = target_base_z - 0.20
        env.cfg.horizon_workspace_z_range = tuple(wz)
        print(f"[GROUND_DEBUG] widened workspace z min to {wz[0]:.3f} for fixed world ground height", flush=True)


def scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.ee_action_mode = "task_space_delta"
cfg.ee_action_dim = int(args_cli.ee_action_dim)
cfg.lowlevel_actuator_backend = str(args_cli.backend)
cfg.ppo_stage = "local_reach_stage0"
cfg.stage0_level = 0
cfg.trajectory_mode = "none"
cfg.trajectory_plane = str(args_cli.trajectory_plane)
cfg.ground_workspace_json = str(args_cli.ground_workspace_json)
cfg.tool_contact_mode = str(args_cli.tool_contact_mode)
cfg.tool_tip_local_offset_x = float(args_cli.tool_tip_local_offset_x)
cfg.tool_tip_local_offset_y = float(args_cli.tool_tip_local_offset_y)
cfg.tool_tip_local_offset_z = float(args_cli.tool_tip_local_offset_z)
cfg.tool_ground_clearance = float(args_cli.tool_ground_clearance)
cfg.min_arm_ground_clearance = float(args_cli.min_arm_ground_clearance)
cfg.ground_z = float(args_cli.ground_z)
cfg.trajectory_scale = float(args_cli.trajectory_scale)
cfg.visualize_target_trajectory = bool(args_cli.visualize_target_trajectory)
cfg.visualize_actual_ee_trail = bool(args_cli.visualize_actual_ee_trail)
cfg.debug_draw_tool_aabb = bool(args_cli.debug_draw_tool_aabb)
cfg.debug_draw_tool_contact_point = bool(args_cli.debug_draw_tool_contact_point)
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
    target_world_z = float(args_cli.ground_z) + float(args_cli.tool_ground_clearance)
    ensure_ground_z_in_workspace(env, target_world_z)
    path = env.configure_ground_trajectory(
        mode=args_cli.trajectory_mode,
        workspace_json=args_cli.ground_workspace_json,
        trajectory_plane=args_cli.trajectory_plane,
        trajectory_scale=args_cli.trajectory_scale,
        visualize_target_trajectory=args_cli.visualize_target_trajectory,
        visualize_actual_ee_trail=args_cli.visualize_actual_ee_trail,
    )
    frame = env.discover_controlled_frame()
    tool_report = env.tool_geometry_report()
    print("[GROUND_DEBUG] controlled frame discovery:", flush=True)
    for key, value in frame.items():
        print(f"  {key}: {value}", flush=True)
    print("[GROUND_DEBUG] tool geometry:", flush=True)
    for key, value in tool_report.items():
        print(f"  {key}: {value}", flush=True)
    if env._ground_limitation:
        print(f"[GROUND_DEBUG] LIMITATION: {env._ground_limitation}", flush=True)
    print(f"[GROUND_DEBUG] path_points={path.shape[0]} params={env._ground_path_params}", flush=True)

    target_clearance_min = float("inf")
    ee_clearance_min = float("inf")
    arm_clearance_min = float("inf")
    max_abs_body = 0.0
    nan_inf_count = 0
    huge_transform_warning_count = 0
    tracking_errors: list[float] = []
    cylinder_limit_values: list[float] = []
    root_motion_max = 0.0
    completion_max = 0.0
    terminated_count = 0
    truncated_count = 0

    for step in range(int(args_cli.num_steps)):
        action = env.compute_ground_tracking_action(gain=1.0)
        _, _, terminated, truncated, extras = env.step(action)
        metrics = env.ground_safety_metrics()
        target_clearance_min = min(target_clearance_min, scalar(metrics["target_tool_bottom_clearance"].min()))
        ee_clearance_min = min(ee_clearance_min, scalar(metrics["tool_bottom_clearance"].min()))
        arm_clearance_min = min(arm_clearance_min, scalar(metrics["arm_geometry_clearance"].min()))
        max_abs_body = max(max_abs_body, scalar(metrics["max_abs_body_world_position"].max()))
        nan_inf_count += int(scalar(metrics["nan_inf_count"].sum()))
        huge_transform_warning_count += int(scalar((metrics["max_abs_body_world_position"] > 100.0).sum()))
        tracking_errors.append(scalar(metrics["tracking_error"].mean()))
        cylinder_limit_values.append(scalar(metrics["cylinder_limit_fraction"].mean()))
        root_motion_max = max(root_motion_max, scalar(metrics["root_motion_norm"].max()))
        completion_max = max(completion_max, scalar(metrics["path_completion_fraction"].max()))
        terminated_count += int(terminated.sum().detach().cpu().item())
        truncated_count += int(truncated.sum().detach().cpu().item())
        if step % 25 == 0:
            print(
                f"[GROUND_DEBUG] step={step:04d} completion={completion_max:.3f} "
                f"err={tracking_errors[-1]:.3f} tool_clear={ee_clearance_min:.3f} arm_geom_clear={arm_clearance_min:.3f}",
                flush=True,
            )
        if completion_max >= 1.0:
            break

    mean_tracking = sum(tracking_errors) / max(len(tracking_errors), 1)
    max_tracking = max(tracking_errors) if tracking_errors else float("inf")
    cylinder_limit_fraction = sum(cylinder_limit_values) / max(len(cylinder_limit_values), 1)
    safety_ok = (
        target_clearance_min >= float(args_cli.tool_ground_clearance) - 0.005
        and ee_clearance_min >= -0.01
        and arm_clearance_min >= -0.02
        and max_abs_body < 100.0
        and nan_inf_count == 0
        and huge_transform_warning_count == 0
        and cylinder_limit_fraction == 0.0
        and terminated_count == 0
        and truncated_count == 0
    )
    completion_ok = completion_max > 0.80 or not bool(args_cli.require_path_completion)
    pass_ok = safety_ok and completion_ok
    print("\nGROUND TRAJECTORY VALIDATION", flush=True)
    print(f"  mode: {args_cli.trajectory_mode}", flush=True)
    print(f"  trajectory visible: {'yes' if args_cli.visualize_target_trajectory else 'no'}", flush=True)
    print(f"  current target marker: {'yes' if args_cli.visualize_target_trajectory else 'no'}", flush=True)
    print(f"  actual EE trail visible: {'yes' if args_cli.visualize_actual_ee_trail else 'no'}", flush=True)
    print(f"  EE body name: {tool_report['ee_body_name']}", flush=True)
    print(f"  EE body path: {tool_report['ee_body_prim_path']}", flush=True)
    print(f"  EE body origin world position: {tool_report['ee_origin_world_position']}", flush=True)
    print(f"  tool/contact point world position: {tool_report['tool_contact_world_position']}", flush=True)
    print(f"  tool bottom offset from EE origin: {tool_report['tool_bottom_offset_from_ee_origin']}", flush=True)
    print(f"  tool_contact_mode: {tool_report['tool_contact_mode']}", flush=True)
    print("  tracking point: tool_bottom_contact", flush=True)
    print(f"  target_world_z_min: {float(path[:, 2].min().detach().cpu().item()):.6f}", flush=True)
    print(f"  target_world_z_max: {float(path[:, 2].max().detach().cpu().item()):.6f}", flush=True)
    print(f"  target_tool_bottom_clearance_min: {target_clearance_min:.6f}", flush=True)
    print(f"  tool_bottom_clearance_min: {ee_clearance_min:.6f}", flush=True)
    print(f"  lowest_non_tool_arm_geometry_clearance_min: {arm_clearance_min:.6f}", flush=True)
    print(f"  max_abs_body_world_position: {max_abs_body:.6f}", flush=True)
    print(f"  NaN/Inf count: {nan_inf_count}", flush=True)
    print(f"  huge transform warning count: {huge_transform_warning_count}", flush=True)
    print(f"  mean tracking error: {mean_tracking:.6f}", flush=True)
    print(f"  max tracking error: {max_tracking:.6f}", flush=True)
    print(f"  path completion fraction: {completion_max:.6f}", flush=True)
    print(f"  require_path_completion: {bool(args_cli.require_path_completion)}", flush=True)
    print(f"  safety_pass: {'yes' if safety_ok else 'no'}", flush=True)
    print(f"  cylinder limit fraction: {cylinder_limit_fraction:.6f}", flush=True)
    print(f"  root motion norm: {root_motion_max:.6f}", flush=True)
    print(f"  terminated_count: {terminated_count}", flush=True)
    print(f"  truncated_count: {truncated_count}", flush=True)
    print(f"  PASS: {'yes' if pass_ok else 'no'}", flush=True)
    if args_cli.fast_exit:
        os._exit(0 if pass_ok else 1)
finally:
    env.close()
    simulation_app.close()
