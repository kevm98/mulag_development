#!/usr/bin/env python3
"""Playback/evaluate horizon-IDFD PPO checkpoint."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--horizon_action_scale_m", type=float, default=0.01)
parser.add_argument("--fd_horizon_gain_per_cylinder", type=str, default="2.0,3.0,3.0,3.0")
parser.add_argument("--checkpoint", type=str, default="", help="RSL-RL checkpoint to load")
parser.add_argument("--episodes", type=int, default=5)
# -- Ground trajectory evaluation: must match the training env exactly --
parser.add_argument(
    "--trajectory_mode",
    type=str,
    default="none",
    choices=("none", "ground_straight_line", "ground_semicircle", "ground_mower_sweep"),
)
parser.add_argument("--trajectory_plane", type=str, default="base_yz", choices=("ground_xy", "base_yz"))
parser.add_argument(
    "--workspace_json",
    type=str,
    default="outputs/ppo1_horizon_idfd/dynamic_hold_workspace.json",
)
parser.add_argument("--ground_max_episode_steps", type=int, default=300)
parser.add_argument("--id_checkpoint", type=str, default="", help="Override PhysX-finetuned ID checkpoint")
parser.add_argument("--ground_path_pacing", choices=("fixed", "capture"), default="fixed")
parser.add_argument("--ground_path_capture_radius_m", type=float, default=0.20)
parser.add_argument("--ground_path_capture_required_hold_steps", type=int, default=3)
parser.add_argument(
    "--ground_initial_target_mode",
    choices=("path_start", "nearest_path", "local_offset"),
    default="local_offset",
)
parser.add_argument("--ground_initial_target_offset_m", type=float, default=0.08)
parser.add_argument("--ground_initial_target_capture_radius_m", type=float, default=0.03)
parser.add_argument("--ground_anchor_path_to_current_tool", type=int, default=1)
parser.add_argument("--ground_anchor_margin_m", type=float, default=0.05)
parser.add_argument("--ground_anchor_max_span_m", type=float, default=0.50)
parser.add_argument("--ground_anchor_static_sag_m", type=float, default=0.03)
parser.add_argument("--reference_space_target", type=int, default=1)
parser.add_argument("--mowing_terrain", type=int, default=0)
parser.add_argument("--mowing_terrain_collision", type=int, default=0)
parser.add_argument("--ee_lam", type=float, default=0.1)
parser.add_argument("--ee_per_cyl_clip", type=float, default=0.003)
parser.add_argument("--ee_global_clip", type=float, default=0.006)
parser.add_argument("--horizon_cmd_clip", type=float, default=0.004)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""): args_cli.device = "cuda:0"
if not args_cli.checkpoint:
    parser.error("--checkpoint is required unless you are using --help")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import (
    MulagUpperLevelEnvCfg,
    MulagUpperLevelPPORunnerCfg,
)
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import (
    HorizonIDFDEnv,
    HorizonIDFDRslRlVecEnvWrapper,
)
from rsl_rl.runners import OnPolicyRunner

fd_gains = [float(x.strip()) for x in args_cli.fd_horizon_gain_per_cylinder.split(",")]
env_cfg = MulagUpperLevelEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
if args_cli.id_checkpoint:
    id_path = Path(args_cli.id_checkpoint).resolve()
    if not id_path.is_file():
        raise RuntimeError(f"ID checkpoint not found at {id_path}")
    env_cfg.id_checkpoint_override_path = str(id_path)
if args_cli.trajectory_mode.startswith("ground_"):
    # Mirror train_upper_level_ppo_horizon_idfd.py exactly — the checkpoint's
    # obs/action dims and gains must match the env it was trained in.
    env_cfg.ppo_stage = "full_mowing"
    env_cfg.trajectory_mode = args_cli.trajectory_mode
    env_cfg.trajectory_plane = args_cli.trajectory_plane
    env_cfg.ground_workspace_json = args_cli.workspace_json
    env_cfg.ground_max_episode_steps = int(args_cli.ground_max_episode_steps)
    env_cfg.ground_path_loop = True
    env_cfg.ground_path_pacing = str(args_cli.ground_path_pacing)
    env_cfg.ground_path_capture_radius_m = float(args_cli.ground_path_capture_radius_m)
    env_cfg.ground_path_capture_required_hold_steps = int(args_cli.ground_path_capture_required_hold_steps)
    env_cfg.ground_initial_target_mode = str(args_cli.ground_initial_target_mode)
    env_cfg.ground_initial_target_offset_m = float(args_cli.ground_initial_target_offset_m)
    env_cfg.ground_initial_target_capture_radius_m = float(args_cli.ground_initial_target_capture_radius_m)
    env_cfg.ground_anchor_path_to_current_tool = bool(args_cli.ground_anchor_path_to_current_tool)
    env_cfg.ground_anchor_margin_m = float(args_cli.ground_anchor_margin_m)
    env_cfg.ground_anchor_max_span_m = float(args_cli.ground_anchor_max_span_m)
    env_cfg.ground_anchor_static_sag_m = float(args_cli.ground_anchor_static_sag_m)
    env_cfg.horizon_reference_space_target = bool(args_cli.reference_space_target)
    env_cfg.mowing_terrain_enable = bool(args_cli.mowing_terrain)
    env_cfg.mowing_terrain_collision = bool(args_cli.mowing_terrain_collision)
    if env_cfg.mowing_terrain_enable:
        try:
            import json as _json
            _ws = _json.load(open(args_cli.workspace_json))
            if "mowing_terrain_z" in _ws:
                env_cfg.mowing_terrain_z = float(_ws["mowing_terrain_z"])
            _sl = _ws.get("base_yz_slice", {})
            if "fixed_world_x" in _sl:
                env_cfg.mowing_terrain_local_x = float(_sl["fixed_world_x"])
            if "y_min" in _sl and "y_max" in _sl:
                env_cfg.mowing_terrain_local_y = 0.5 * (float(_sl["y_min"]) + float(_sl["y_max"]))
                env_cfg.mowing_terrain_half_y = max(0.5 * (float(_sl["y_max"]) - float(_sl["y_min"])) + 0.15, 0.3)
        except Exception:
            pass
    env_cfg.ground_hold_target = False
    env_cfg.ee_action_dim = 2 if args_cli.trajectory_plane in ("base_yz", "ground_xy") else 3
    env_cfg.ee_k_p = 0.15
    env_cfg.ee_lam = float(args_cli.ee_lam)
    env_cfg.ee_per_cyl_clip_m = float(args_cli.ee_per_cyl_clip)
    env_cfg.ee_global_clip_m = float(args_cli.ee_global_clip)
    env_cfg.ee_smooth_alpha = 0.7
    env_cfg.horizon_delta_cmd_clip_m = float(args_cli.horizon_cmd_clip)
    env_cfg.horizon_reset_settle_steps = 320
    env_cfg.tool_contact_mode = "geometry_aabb"
    env_cfg.visualize_target_trajectory = False
    env_cfg.visualize_actual_ee_trail = False
    env_cfg.debug_draw_tool_contact_point = False
    env_cfg.debug_draw_tool_aabb = False
if getattr(args_cli, "device", None): env_cfg.device = args_cli.device; env_cfg.sim.device = args_cli.device

base_env = HorizonIDFDEnv(
    base_cfg=env_cfg,
    horizon_steps=args_cli.horizon_steps,
    action_scale_m=args_cli.horizon_action_scale_m,
    fd_gains=fd_gains,
    device=args_cli.device,
)
env = HorizonIDFDRslRlVecEnvWrapper(base_env)

# Build runner and load checkpoint
agent_cfg = MulagUpperLevelPPORunnerCfg()
agent_cfg.device = args_cli.device
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="/tmp/horizon_play", device=args_cli.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=args_cli.device)
print(f"Loaded checkpoint: {args_cli.checkpoint}", flush=True)

max_ep_steps = (
    int(args_cli.ground_max_episode_steps)
    if args_cli.trajectory_mode.startswith("ground_")
    else 100
)
obs, _ = env.reset()
for ep in range(args_cli.episodes):
    total_rew = 0.0
    err_sum = 0.0
    err_last = float("nan")
    n_steps = 0
    for step in range(max_ep_steps):
        with torch.no_grad():
            action = policy(obs)
        obs, rew, dones, _ = env.step(action)
        total_rew += float(rew.mean().cpu())
        n_steps += 1
        if args_cli.trajectory_mode.startswith("ground_"):
            err_last = float(base_env.ground_safety_metrics()["tracking_error"].mean().cpu())
            err_sum += err_last
        if dones.any():
            break
    mean_err = err_sum / max(n_steps, 1)
    print(
        f"Episode {ep+1}: steps={n_steps} total_reward={total_rew:.2f} "
        f"mean_tracking_error={mean_err:.3f} final_tracking_error={err_last:.3f}",
        flush=True,
    )

env.close()
simulation_app.close()
