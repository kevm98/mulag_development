#!/usr/bin/env python3
"""Train horizon-displacement PPO with ID/FD closed-loop pipeline."""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime
from pathlib import Path
import pickle
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Horizon-IDFD PPO training.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_iterations", type=int, default=200)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--ee_action_mode", type=str, default="task_space_delta")
parser.add_argument("--ee_action_scale_m", type=float, default=0.05)
parser.add_argument("--horizon_action_scale_m", type=float, default=0.01)
parser.add_argument("--fd_horizon_gain_per_cylinder", type=str, default="2.0,3.0,3.0,3.0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--stage0_level", type=int, default=0)
parser.add_argument("--ppo_stage", type=str, default="local_reach_stage0")
# -- Ground trajectory tracking training (overrides ppo_stage to full_mowing) --
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
    help="Ground target used at reset before following the full path.",
)
parser.add_argument("--ground_initial_target_offset_m", type=float, default=0.08)
parser.add_argument("--ground_initial_target_capture_radius_m", type=float, default=0.03)
parser.add_argument("--ground_anchor_path_to_current_tool", type=int, default=1)
parser.add_argument("--ground_anchor_margin_m", type=float, default=0.05)
parser.add_argument("--ground_anchor_max_span_m", type=float, default=0.50)
parser.add_argument("--ground_anchor_static_sag_m", type=float, default=0.03)
# Reference-space command integration (target += dx) instead of re-anchoring to the
# sagged measured position each step. 1 = on (fixes the gravity ratchet that made
# upward-Z tracking impossible; proven by axis_probe). 0 = legacy measured-space.
parser.add_argument("--reference_space_target", type=int, default=1)
parser.add_argument("--mowing_terrain", type=int, default=0, help="1 = spawn per-env mowable shelf at the tool low band")
parser.add_argument("--mowing_terrain_collision", type=int, default=0)
parser.add_argument("--ee_lam", type=float, default=0.1)
parser.add_argument("--ee_per_cyl_clip", type=float, default=0.003)
parser.add_argument("--ee_global_clip", type=float, default=0.006)
parser.add_argument("--horizon_cmd_clip", type=float, default=0.004)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""): args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np, torch
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg, MulagUpperLevelPPORunnerCfg
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import (
    HorizonIDFDEnv,
    HorizonIDFDRslRlVecEnvWrapper,
)
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml


def dump_pickle(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(data, stream)

# ---- SAFETY GUARDS ----
env_cfg = MulagUpperLevelEnvCfg()
if env_cfg.low_level_servo_mode == "position_ramp_pi":
    raise RuntimeError("REFUSING: position_ramp_pi is diagnostic only, not for PPO training.")
if not env_cfg.use_sequence_cylinder_models:
    raise RuntimeError("REFUSING: ID/FD must be enabled for horizon PPO.")

id_path = (
    Path(args_cli.id_checkpoint).resolve()
    if args_cli.id_checkpoint
    else (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
)
if not id_path.is_file():
    raise RuntimeError(f"Fine-tuned ID checkpoint not found at {id_path}")

fd_gains = [float(x.strip()) for x in args_cli.fd_horizon_gain_per_cylinder.split(",")]
if len(fd_gains) != 4:
    raise RuntimeError(f"fd_horizon_gain_per_cylinder must have 4 values, got {len(fd_gains)}")

print(f"SAFETY CHECKS PASSED", flush=True)
print(f"  ID checkpoint: {id_path}", flush=True)
print(f"  fd_gains: {fd_gains}", flush=True)
print(f"  horizon_steps: {args_cli.horizon_steps}", flush=True)
print(f"  action_scale: {args_cli.horizon_action_scale_m}m", flush=True)

# ---- Build env ----
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.id_checkpoint_override_path = str(id_path)
env_cfg.ppo_stage = args_cli.ppo_stage
env_cfg.stage0_level = args_cli.stage0_level
env_cfg.ee_action_mode = args_cli.ee_action_mode
env_cfg.ee_action_scale_m = args_cli.ee_action_scale_m
if args_cli.trajectory_mode.startswith("ground_"):
    # Trajectory-tracking stage: targets come from the dynamic hold workspace and
    # advance along the path each step; mirror the validated observe-mode setup.
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
apply_kevin_sim_memory_optimizations(env_cfg.sim, num_envs=args_cli.num_envs, verbose=True)

base_env = HorizonIDFDEnv(
    base_cfg=env_cfg,
    horizon_steps=args_cli.horizon_steps,
    action_scale_m=args_cli.horizon_action_scale_m,
    fd_gains=fd_gains,
    device=args_cli.device,
)

env = HorizonIDFDRslRlVecEnvWrapper(base_env)
print(f"Env wrapped: {args_cli.num_envs} envs, obs={base_env.observation_space}, act={base_env.num_actions}", flush=True)

# ---- Build PPO runner using existing config ----
agent_cfg = MulagUpperLevelPPORunnerCfg()
agent_cfg.max_iterations = args_cli.max_iterations
agent_cfg.device = args_cli.device
agent_cfg.seed = 42

# Log directory
log_root = os.path.join("logs", "rsl_rl", "horizon_idfd_ppo")
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(log_root, run_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
print(f"Log dir: {log_dir}", flush=True)

# Save horizon config
try:
    import yaml as _yaml
    _yaml.dump({"horizon_steps": args_cli.horizon_steps,
                "action_scale": args_cli.horizon_action_scale_m,
                "fd_gains": fd_gains,
                "num_envs": args_cli.num_envs,
                "max_iterations": args_cli.max_iterations,
                "stage0_level": args_cli.stage0_level,
                "ppo_stage": args_cli.ppo_stage,
                "trajectory_mode": args_cli.trajectory_mode,
                "trajectory_plane": args_cli.trajectory_plane,
                "ground_path_pacing": args_cli.ground_path_pacing,
                "ground_path_capture_radius_m": args_cli.ground_path_capture_radius_m,
                "ground_initial_target_mode": args_cli.ground_initial_target_mode,
                "ground_initial_target_offset_m": args_cli.ground_initial_target_offset_m,
                "ground_initial_target_capture_radius_m": args_cli.ground_initial_target_capture_radius_m,
                "ee_per_cyl_clip": args_cli.ee_per_cyl_clip,
                "ee_global_clip": args_cli.ee_global_clip,
                "horizon_cmd_clip": args_cli.horizon_cmd_clip,
                "id_checkpoint": str(id_path)},
               open(os.path.join(log_dir, "params", "horizon_config.yaml"), "w"))
except Exception:
    pass

# Create runner using agent_cfg.to_dict() — same pattern as reference training script
runner = OnPolicyRunner(
    env,
    agent_cfg.to_dict(),
    log_dir=log_dir,
    device=agent_cfg.device,
)

if args_cli.checkpoint:
    print(f"Resuming from checkpoint: {args_cli.checkpoint}", flush=True)
    runner.load(args_cli.checkpoint)

dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

print(f"Starting PPO training: {args_cli.max_iterations} iterations...", flush=True)
try:
    with torch.inference_mode(False):
        runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    print("PPO training finished; closing simulation.", flush=True)
finally:
    env.close()
    simulation_app.close()
    print("Training complete.", flush=True)
