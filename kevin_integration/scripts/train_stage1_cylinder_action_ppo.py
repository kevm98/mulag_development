#!/usr/bin/env python3
"""No-Jacobian Stage-1 PPO — policy outputs 4D cylinder actions directly."""
from __future__ import annotations
import argparse, os, sys, json
from datetime import datetime
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="No-Jacobian cylinder-action PPO training.")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--max_iterations", type=int, default=50)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--num_steps_per_env", type=int, default=8)
parser.add_argument("--save_interval", type=int, default=5)
parser.add_argument("--trajectory_mode", type=str, default="sequence_easy")
parser.add_argument("--target_hold_steps", type=int, default=50)
parser.add_argument("--workspace_json", type=str, default=None)
parser.add_argument("--target_y_range", type=float, default=None)
parser.add_argument("--target_z_range", type=float, default=None)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--bias_calibration_time_s", type=float, default=5.0)
parser.add_argument("--post_reset_settle_steps", type=int, default=5)
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.0025)
parser.add_argument("--action_smoothing_beta", type=float, default=0.8)
parser.add_argument("--action_rate_limit", type=float, default=0.1)
parser.add_argument("--reward_mode", type=str, default="distance_only")
parser.add_argument("--experiment_name", type=str, default="ppo1_nojac")
parser.add_argument("--ppo_lr_scale", type=float, default=0.1)
parser.add_argument("--ppo_clip_param", type=float, default=0.05)
parser.add_argument("--ppo_entropy_coef", type=float, default=0.001)
parser.add_argument("--ppo_num_learning_epochs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1_nojac.stage1_cylinder_action_env_wrapper import Stage1CylinderActionEnvWrapper
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg
from rsl_rl.runners import OnPolicyRunner

# Load workspace
ws = {}
if args_cli.workspace_json and Path(args_cli.workspace_json).exists():
    with open(args_cli.workspace_json) as f:
        ws = json.load(f)
    print(f"[INFO] Loaded workspace: Y=[{ws.get('safe_y_min',0):.3f},{ws.get('safe_y_max',0):.3f}] "
          f"Z=[{ws.get('safe_z_min',0):.3f},{ws.get('safe_z_max',0):.3f}]", flush=True)

print(f"[INFO] No-Jacobian cylinder-action PPO", flush=True)
print(f"[INFO] obs_dim=18 act_dim=4 trajectory={args_cli.trajectory_mode}", flush=True)

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
env = Stage1CylinderActionEnvWrapper(cfg,
    horizon_steps=int(args_cli.horizon_steps),
    device=args_cli.device,
    kp_scale=float(args_cli.kp_scale),
    kd_scale=float(args_cli.kd_scale),
    trajectory_mode=args_cli.trajectory_mode,
    target_hold_steps=int(args_cli.target_hold_steps),
    post_reset_settle_steps=int(args_cli.post_reset_settle_steps),
    backend=str(args_cli.backend),
    bias_calibration_time_s=float(args_cli.bias_calibration_time_s),
    max_cylinder_delta_m=float(args_cli.max_cylinder_delta_m),
    action_smoothing_beta=float(args_cli.action_smoothing_beta),
    action_rate_limit=float(args_cli.action_rate_limit),
    reward_mode=str(args_cli.reward_mode),
    workspace_yz=ws)

B = env.num_envs
device = args_cli.device

if args_cli.target_y_range is not None:
    env._target_y_range = float(args_cli.target_y_range)
if args_cli.target_z_range is not None:
    env._target_z_range = float(args_cli.target_z_range)

# Override for easy/normal modes based on workspace
if ws:
    if 'easy' in args_cli.trajectory_mode:
        env._target_y_range = ws.get("recommended_target_y_range", 0.03) * 0.5
        env._target_z_range = ws.get("recommended_target_z_range", 0.01) * 0.5
    elif 'normal' in args_cli.trajectory_mode:
        env._target_y_range = ws.get("recommended_target_y_range", 0.03) * 0.8
        env._target_z_range = ws.get("recommended_target_z_range", 0.01) * 0.8

print(f"[INFO] target_y_range={env._target_y_range:.4f} target_z_range={env._target_z_range:.4f}", flush=True)

log_root = os.path.join("logs", "rsl_rl", args_cli.experiment_name)
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(log_root, run_name)
os.makedirs(log_dir, exist_ok=True)
print(f"[INFO] log_dir = {log_dir}", flush=True)

# Gate A: zero hold
print(f"\n{'='*60}", flush=True)
print(f"GATE A: zero-action hold", flush=True)
print(f"{'='*60}", flush=True)
env.reset()
for _ in range(10):
    env.step(torch.zeros((B, 4), device=device))
jumps = []
for _ in range(20):
    _, _, _, extras = env.step(torch.zeros((B, 4), device=device))
    jumps.append(extras["log"]["ee_jump"].item())
mean_j = float(np.mean(jumps)); max_j = float(np.max(jumps))
print(f"  mean_jump = {mean_j:.4f}m  max_jump = {max_j:.4f}m", flush=True)
gate_a = (mean_j < 0.06 and max_j < 0.10)
print(f"  GATE A: {'PASS' if gate_a else 'FAIL'}", flush=True)
if not gate_a:
    print("[ABORT] Gate A failed.", flush=True)
    env.close(); simulation_app.close(); sys.exit(1)

# Build runner
agent_cfg = MulagUpperLevelPPORunnerCfg()
agent_cfg.max_iterations = int(args_cli.max_iterations)
agent_cfg.num_steps_per_env = int(args_cli.num_steps_per_env)
agent_cfg.save_interval = int(args_cli.save_interval)
agent_cfg.device = device; agent_cfg.seed = 42
agent_cfg.actor.hidden_dims = [128, 64]; agent_cfg.critic.hidden_dims = [128, 64]
agent_cfg.algorithm.learning_rate = 3.0e-4 * float(args_cli.ppo_lr_scale)
if float(args_cli.ppo_clip_param) > 0:
    agent_cfg.algorithm.clip_param = float(args_cli.ppo_clip_param)
if float(args_cli.ppo_entropy_coef) >= 0:
    agent_cfg.algorithm.entropy_coef = float(args_cli.ppo_entropy_coef)
if int(args_cli.ppo_num_learning_epochs) > 0:
    agent_cfg.algorithm.num_learning_epochs = int(args_cli.ppo_num_learning_epochs)
agent_cfg.actor.obs_normalization = False
agent_cfg.critic.obs_normalization = False
agent_cfg.experiment_name = args_cli.experiment_name

print(f"[INFO] PPO: lr={agent_cfg.algorithm.learning_rate:.2e} clip={getattr(agent_cfg.algorithm,'clip_param','def')} "
      f"entropy={agent_cfg.algorithm.entropy_coef} epochs={getattr(agent_cfg.algorithm,'num_learning_epochs','def')} "
      f"max_delta={args_cli.max_cylinder_delta_m}", flush=True)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

# Zero-initialize actor output layer so policy starts near-zero (safe hold)
with torch.no_grad():
    runner.alg.actor.mlp[-1].weight.zero_()
    runner.alg.actor.mlp[-1].bias.zero_()
print("[INFO] Zero-initialized actor output layer", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"[INFO] Starting training ({args_cli.max_iterations} iters)", flush=True)
print(f"{'='*60}", flush=True)

try:
    with torch.inference_mode(False):
        runner.learn(num_learning_iterations=int(args_cli.max_iterations), init_at_random_ep_len=True)
    print("\n[INFO] Training complete", flush=True)
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}", flush=True)
    import traceback; traceback.print_exc()

final_path = os.path.join(log_dir, "model_final.pt")
try:
    sd = runner.alg.save(); sd["iter"] = runner.current_learning_iteration; sd["infos"] = {}
    torch.save(sd, final_path)
    print(f"[INFO] Saved: {final_path}", flush=True)
except Exception as e:
    print(f"[WARN] Save failed: {e}", flush=True)

env.close()
simulation_app.close()
print("Done.", flush=True)
