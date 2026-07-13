#!/usr/bin/env python3
"""Stage-1 Y-Z residual PPO — base greedy + PPO additive correction."""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage-1 Y-Z residual PPO training.")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--max_iterations", type=int, default=50)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--action_scale_m", type=float, default=0.05)
parser.add_argument("--num_steps_per_env", type=int, default=8)
parser.add_argument("--save_interval", type=int, default=5)
parser.add_argument("--trajectory_mode", type=str, default="sequence")
parser.add_argument("--target_hold_steps", type=int, default=50)
parser.add_argument("--target_y_range", type=float, default=0.03)
parser.add_argument("--target_z_range", type=float, default=0.01)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--bias_calibration_time_s", type=float, default=5.0)
parser.add_argument("--post_reset_settle_steps", type=int, default=5)
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--force_rate_limit_fraction", type=float, default=0.01)
parser.add_argument("--velocity_filter_beta", type=float, default=0.8)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.005)
parser.add_argument("--experiment_name", type=str, default="stage1_residual")
parser.add_argument("--reward_mode", type=str, default="distance_only")
parser.add_argument("--action_penalty_weight", type=float, default=0.005)
parser.add_argument("--ee_jump_penalty_weight", type=float, default=0.05)
parser.add_argument("--action_smoothing_beta", type=float, default=0.8)
parser.add_argument("--action_rate_limit", type=float, default=0.1)
parser.add_argument("--base_gain", type=float, default=1.0)
parser.add_argument("--residual_scale", type=float, default=0.05)
parser.add_argument("--ppo_lr_scale", type=float, default=0.03)
parser.add_argument("--ppo_clip_param", type=float, default=0.03)
parser.add_argument("--ppo_entropy_coef", type=float, default=0.0)
parser.add_argument("--ppo_num_learning_epochs", type=int, default=1)
parser.add_argument("--zero_init_output", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1_residual.stage1_yz_residual_env_wrapper import Stage1YZResidualEnvWrapper
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg
from rsl_rl.runners import OnPolicyRunner

print(f"[INFO] Stage-1 Y-Z residual PPO", flush=True)
print(f"[INFO] trajectory_mode = {args_cli.trajectory_mode}", flush=True)
print(f"[INFO] obs_dim = 8, act_dim = 2 (residual)", flush=True)
print(f"[INFO] base_gain = {args_cli.base_gain}, residual_scale = {args_cli.residual_scale}", flush=True)
print(f"[INFO] kp_scale = {args_cli.kp_scale}, kd_scale = {args_cli.kd_scale}", flush=True)

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
env = Stage1YZResidualEnvWrapper(cfg,
                                  horizon_steps=int(args_cli.horizon_steps),
                                  action_scale_m=float(args_cli.action_scale_m),
                                  device=args_cli.device,
                                  kp_scale=float(args_cli.kp_scale),
                                  kd_scale=float(args_cli.kd_scale),
                                  force_rate_limit=float(args_cli.force_rate_limit_fraction),
                                  velocity_filter_beta=float(args_cli.velocity_filter_beta),
                                  trajectory_mode=args_cli.trajectory_mode,
                                  target_hold_steps=int(args_cli.target_hold_steps),
                                  post_reset_settle_steps=int(args_cli.post_reset_settle_steps),
                                  backend=str(args_cli.backend),
                                  bias_calibration_time_s=float(args_cli.bias_calibration_time_s),
                                  reward_mode=str(args_cli.reward_mode),
                                  action_penalty_weight=float(args_cli.action_penalty_weight),
                                  ee_jump_penalty_weight=float(args_cli.ee_jump_penalty_weight),
                                  action_smoothing_beta=float(args_cli.action_smoothing_beta),
                                  action_rate_limit=float(args_cli.action_rate_limit),
                                  base_gain=float(args_cli.base_gain),
                                  residual_scale=float(args_cli.residual_scale))
B = env.num_envs
device = args_cli.device
env._target_y_range = float(args_cli.target_y_range)
env._target_z_range = float(args_cli.target_z_range)

log_root = os.path.join("logs", "rsl_rl", args_cli.experiment_name)
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(log_root, run_name)
os.makedirs(log_dir, exist_ok=True)
print(f"[INFO] log_dir = {log_dir}", flush=True)

# Gate A: zero hold
print(f"\n{'='*60}", flush=True)
print(f"GATE A: zero-action no-pin hold", flush=True)
print(f"{'='*60}", flush=True)
env.reset()
for _ in range(10):
    env.step(torch.zeros((B, 2), device=device))
jumps = []
for _ in range(20):
    _, _, _, extras = env.step(torch.zeros((B, 2), device=device))
    jumps.append(extras["log"]["ee_jump"].item())
mean_j = float(np.mean(jumps)); max_j = float(np.max(jumps))
print(f"  mean_jump = {mean_j:.4f} m  max_jump = {max_j:.4f} m", flush=True)
gate_a = (mean_j < 0.06 and max_j < 0.10)
print(f"  GATE A: {'PASS' if gate_a else 'FAIL'}", flush=True)
if not gate_a:
    print("[ABORT] Gate A failed.", flush=True)
    env.close(); simulation_app.close(); sys.exit(1)

# Gate B: base controller sanity
print(f"\n{'='*60}", flush=True)
print(f"GATE B: Base controller (residual=0) sanity", flush=True)
print(f"{'='*60}", flush=True)
# Force residual to zero
orig_rs = env._residual_scale
env._residual_scale = 0.0
env.reset()
for _ in range(30):
    env.step(torch.zeros((B, 2), device=device))
dists = []
for _ in range(20):
    _, _, _, extras = env.step(torch.zeros((B, 2), device=device))
    dists.append(extras["log"]["distance"].item())
base_dist = float(np.mean(dists))
env._residual_scale = orig_rs
print(f"  base mean_distance = {base_dist:.4f} m", flush=True)
print(f"  GATE B: {'PASS' if base_dist < 0.12 else 'FAIL'}", flush=True)

# Build runner
agent_cfg = MulagUpperLevelPPORunnerCfg()
agent_cfg.max_iterations = int(args_cli.max_iterations)
agent_cfg.num_steps_per_env = int(args_cli.num_steps_per_env)
agent_cfg.save_interval = int(args_cli.save_interval)
agent_cfg.device = device; agent_cfg.seed = 42
agent_cfg.actor.hidden_dims = [64, 32]; agent_cfg.critic.hidden_dims = [64, 32]
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

print(f"[INFO] PPO config: lr={agent_cfg.algorithm.learning_rate:.2e}  "
      f"clip_param={getattr(agent_cfg.algorithm, 'clip_param', 'default')}  "
      f"entropy_coef={agent_cfg.algorithm.entropy_coef}  "
      f"num_learning_epochs={getattr(agent_cfg.algorithm, 'num_learning_epochs', 'default')}  "
      f"residual_scale={args_cli.residual_scale}", flush=True)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

# Zero-initialize actor output layer so residual starts at zero
if args_cli.zero_init_output:
    print(f"[INFO] Zero-initializing actor output layer", flush=True)
    with torch.no_grad():
        runner.alg.actor.mlp[4].weight.zero_()
        runner.alg.actor.mlp[4].bias.zero_()

# Save initial checkpoint
init_path = os.path.join(log_dir, "model_0.pt")
sd = runner.alg.save(); sd["iter"] = 0; sd["infos"] = {}
torch.save(sd, init_path)
print(f"[INFO] Saved init: {init_path}", flush=True)

# PPO training
print(f"\n{'='*60}", flush=True)
print(f"[INFO] Starting residual PPO training", flush=True)
print(f"[INFO] max_iterations = {args_cli.max_iterations}", flush=True)
print(f"[INFO] num_envs = {B}", flush=True)
print(f"{'='*60}", flush=True)

try:
    with torch.inference_mode(False):
        runner.learn(num_learning_iterations=int(args_cli.max_iterations), init_at_random_ep_len=True)
    print("\n[INFO] Residual PPO training complete", flush=True)
except Exception as e:
    print(f"\n[ERROR] PPO failed: {e}", flush=True)
    import traceback; traceback.print_exc()

# Save final
final_path = os.path.join(log_dir, "model_final.pt")
try:
    sd = runner.alg.save(); sd["iter"] = runner.current_learning_iteration; sd["infos"] = {}
    torch.save(sd, final_path)
    print(f"[INFO] Saved final: {final_path}", flush=True)
except Exception as e:
    print(f"[WARN] Save final failed: {e}", flush=True)

env.close()
simulation_app.close()
print("Done.", flush=True)
