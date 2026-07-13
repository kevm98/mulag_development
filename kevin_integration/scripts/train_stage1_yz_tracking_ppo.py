#!/usr/bin/env python3
"""Stage-1 Y-Z trajectory tracking PPO — moving targets, no _pin_root."""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage-1 Y-Z trajectory tracking PPO.")
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--action_scale_m", type=float, default=0.05)
parser.add_argument("--num_steps_per_env", type=int, default=8)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--trajectory_mode", type=str, default="mixed")
parser.add_argument("--target_hold_steps", type=int, default=30)
parser.add_argument("--target_y_range", type=float, default=0.05)
parser.add_argument("--target_z_range", type=float, default=0.02)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--bias_calibration_time_s", type=float, default=2.0)
parser.add_argument("--post_reset_settle_steps", type=int, default=30)
parser.add_argument("--debug_bias_shapes", action="store_true", default=False)
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--force_rate_limit_fraction", type=float, default=0.01)
parser.add_argument("--velocity_filter_beta", type=float, default=0.8)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.005)
parser.add_argument("--experiment_name", type=str, default="stage1_yz_tracking")
parser.add_argument("--stage0_checkpoint", type=str, default=None)
parser.add_argument("--transfer_output_scale", type=float, default=0.25)
parser.add_argument("--action_penalty_weight", type=float, default=0.005)
parser.add_argument("--eval_after_train", action="store_true", default=False)
parser.add_argument("--reward_mode", type=str, default="default")
parser.add_argument("--ee_jump_penalty_weight", type=float, default=0.05)
parser.add_argument("--action_smoothing_beta", type=float, default=0.0)
parser.add_argument("--action_rate_limit", type=float, default=-1.0)
parser.add_argument("--disable_stage0_transfer", action="store_true", default=False)
parser.add_argument("--ppo_lr_scale", type=float, default=1.0)
parser.add_argument("--ppo_clip_param", type=float, default=-1.0)
parser.add_argument("--ppo_entropy_coef", type=float, default=-1.0)
parser.add_argument("--ppo_num_learning_epochs", type=int, default=-1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage1_yz_tracking_env_wrapper import Stage1YZTrackingEnvWrapper
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg
from rsl_rl.runners import OnPolicyRunner

# --- Config ---
print(f"[INFO] Stage-1 Y-Z trajectory tracking", flush=True)
print(f"[INFO] trajectory_mode = {args_cli.trajectory_mode}", flush=True)
print(f"[INFO] obs_dim = 6, act_dim = 2", flush=True)
print(f"[INFO] kp_scale = {args_cli.kp_scale}, kd_scale = {args_cli.kd_scale}", flush=True)
print(f"[INFO] _pin_root disabled", flush=True)

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
env = Stage1YZTrackingEnvWrapper(cfg,
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
                                  debug_bias_shapes=bool(args_cli.debug_bias_shapes),
                                  reward_mode=str(args_cli.reward_mode),
                                  action_penalty_weight=float(args_cli.action_penalty_weight),
                                  ee_jump_penalty_weight=float(args_cli.ee_jump_penalty_weight),
                                  action_smoothing_beta=float(args_cli.action_smoothing_beta),
                                  action_rate_limit=float(args_cli.action_rate_limit))
B = env.num_envs
device = args_cli.device
env._target_y_range = float(args_cli.target_y_range)
env._target_z_range = float(args_cli.target_z_range)

log_root = os.path.join("logs", "rsl_rl", args_cli.experiment_name)
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(log_root, run_name); os.makedirs(log_dir, exist_ok=True)
print(f"[INFO] log_dir = {log_dir}", flush=True)

# ====================================================================
# GATE A: zero-action no-pin hold
# ====================================================================
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

# ====================================================================
# Build runner
# ====================================================================
agent_cfg = MulagUpperLevelPPORunnerCfg()
agent_cfg.max_iterations = int(args_cli.max_iterations)
agent_cfg.num_steps_per_env = int(args_cli.num_steps_per_env)
agent_cfg.save_interval = int(args_cli.save_interval)
agent_cfg.device = device; agent_cfg.seed = 42
agent_cfg.actor.hidden_dims = [64, 32]; agent_cfg.critic.hidden_dims = [64, 32]
agent_cfg.algorithm.learning_rate = 3.0e-4
agent_cfg.algorithm.entropy_coef = 0.001
agent_cfg.actor.obs_normalization = False
agent_cfg.critic.obs_normalization = False
agent_cfg.experiment_name = args_cli.experiment_name
agent_cfg.algorithm.learning_rate = 3.0e-4 * float(args_cli.ppo_lr_scale)
if float(args_cli.ppo_clip_param) > 0:
    agent_cfg.algorithm.clip_param = float(args_cli.ppo_clip_param)
if float(args_cli.ppo_entropy_coef) >= 0:
    agent_cfg.algorithm.entropy_coef = float(args_cli.ppo_entropy_coef)
if int(args_cli.ppo_num_learning_epochs) > 0:
    agent_cfg.algorithm.num_learning_epochs = int(args_cli.ppo_num_learning_epochs)
print(f"[INFO] PPO config: lr={agent_cfg.algorithm.learning_rate:.2e}  "
      f"clip_param={getattr(agent_cfg.algorithm, 'clip_param', 'default')}  "
      f"entropy_coef={agent_cfg.algorithm.entropy_coef}  "
      f"num_learning_epochs={getattr(agent_cfg.algorithm, 'num_learning_epochs', 'default')}", flush=True)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

# ====================================================================
# GATE B: Stage-0 weight transfer
# ====================================================================
if args_cli.stage0_checkpoint and Path(args_cli.stage0_checkpoint).exists() and not args_cli.disable_stage0_transfer:
    print(f"\n{'='*60}", flush=True)
    print(f"GATE B: Stage-0 weight transfer", flush=True)
    print(f"{'='*60}", flush=True)
    s0_ckpt = torch.load(args_cli.stage0_checkpoint, map_location='cpu', weights_only=False)
    s0_sd = s0_ckpt["actor_state_dict"]
    s1_actor = runner.alg.actor
    s1_sd = s1_actor.state_dict()

    # Partial transfer: map Stage-0 obs [err_y, err_z, prev_a_y, prev_a_z]
    # to Stage-1 obs [err_y, err_z, tvel_y, tvel_z, prev_a_y, prev_a_z]
    # Handles different hidden sizes: copy first min(rows_s0, rows_s1) rows
    s0_nh = s0_sd["mlp.0.weight"].shape[0]  # e.g. 64
    s1_nh = s1_sd["mlp.0.weight"].shape[0]  # e.g. 128
    copy_rows = min(s0_nh, s1_nh)
    w0 = s1_sd["mlp.0.weight"].clone()
    w0[:copy_rows, 0] = s0_sd["mlp.0.weight"][:copy_rows, 0]  # err_y
    w0[:copy_rows, 1] = s0_sd["mlp.0.weight"][:copy_rows, 1]  # err_z
    w0[:copy_rows, 4] = s0_sd["mlp.0.weight"][:copy_rows, 2]  # prev_action_y
    w0[:copy_rows, 5] = s0_sd["mlp.0.weight"][:copy_rows, 3]  # prev_action_z
    w0[:copy_rows, 2] = 0.0  # target_vel_y (zero init)
    w0[:copy_rows, 3] = 0.0  # target_vel_z (zero init)
    s1_sd["mlp.0.weight"] = w0
    s1_sd["mlp.0.bias"][:copy_rows] = s0_sd["mlp.0.bias"][:copy_rows].clone()
    # Hidden layer: copy if shapes match, else copy subset
    if s0_sd["mlp.2.weight"].shape == s1_sd["mlp.2.weight"].shape:
        s1_sd["mlp.2.weight"] = s0_sd["mlp.2.weight"].clone()
        s1_sd["mlp.2.bias"] = s0_sd["mlp.2.bias"].clone()
    else:
        s0_h2 = s0_sd["mlp.2.weight"].shape
        s1_h2 = s1_sd["mlp.2.weight"].shape
        cr = min(s0_h2[0], s1_h2[0]); cc = min(s0_h2[1], s1_h2[1])
        s1_sd["mlp.2.weight"][:cr, :cc] = s0_sd["mlp.2.weight"][:cr, :cc].clone()
        s1_sd["mlp.2.bias"][:cr] = s0_sd["mlp.2.bias"][:cr].clone()
    # Output layer: copy subset
    s0_out = s0_sd["mlp.4.weight"].shape
    s1_out = s1_sd["mlp.4.weight"].shape
    cr_o = min(s0_out[0], s1_out[0]); cc_o = min(s0_out[1], s1_out[1])
    s1_sd["mlp.4.weight"][:cr_o, :cc_o] = s0_sd["mlp.4.weight"][:cr_o, :cc_o].clone()
    s1_sd["mlp.4.bias"][:cr_o] = s0_sd["mlp.4.bias"][:cr_o].clone()

    s1_actor.load_state_dict(s1_sd, strict=True)
    print(f"  Stage-0 weights transferred (partial first layer)", flush=True)

    # Cosine check (use subset for weight, full for same-shape layers)
    dev = s1_actor.mlp[0].weight.device
    cosines = []
    # First layer: compare only the copied subset [0:copy_rows, mapped cols]
    cr_w = min(s0_sd["mlp.0.weight"].shape[0], s1_sd["mlp.0.weight"].shape[0])
    s0_sub = s0_sd["mlp.0.weight"][:cr_w].flatten().float().to(dev)
    # Compare s1 subset: cols 0,1 (err), 4,5 (prev_action)
    s1_cols = torch.cat([s1_sd["mlp.0.weight"][:cr_w, 0].flatten(),
                          s1_sd["mlp.0.weight"][:cr_w, 1].flatten(),
                          s1_sd["mlp.0.weight"][:cr_w, 4].flatten(),
                          s1_sd["mlp.0.weight"][:cr_w, 5].flatten()]).float()
    if len(s1_cols) > 0 and len(s0_sub) > 0:
        c = torch.nn.functional.cosine_similarity(s0_sub, s1_cols, dim=0).item()
        cosines.append(c)
    # Output layer: compare subset
    cr_o = min(s0_sd["mlp.4.weight"].shape[0], s1_sd["mlp.4.weight"].shape[0])
    cc_o = min(s0_sd["mlp.4.weight"].shape[1], s1_sd["mlp.4.weight"].shape[1])
    s0_o = s0_sd["mlp.4.weight"][:cr_o, :cc_o].flatten().float().to(dev)
    s1_o = s1_sd["mlp.4.weight"][:cr_o, :cc_o].flatten().float()
    c_o = torch.nn.functional.cosine_similarity(s0_o, s1_o, dim=0).item()
    cosines.append(c_o)
    mean_cos = float(np.mean(cosines)) if cosines else 0.0
    print(f"  Transfer cosine: {mean_cos:.4f}", flush=True)
    gate_b = mean_cos > 0.8
    print(f"  GATE B: {'PASS' if gate_b else 'WARN'}", flush=True)

    # Save transfer checkpoint
    transfer_path = os.path.join(log_dir, "model_stage0_transfer.pt")
    sd = runner.alg.save(); sd["iter"] = 0; sd["infos"] = {}
    torch.save(sd, transfer_path)
    # Apply output scaling to reduce action magnitude
    scale = float(args_cli.transfer_output_scale)
    with torch.no_grad():
        s1_actor.mlp[4].weight *= scale
        s1_actor.mlp[4].bias *= scale
    print(f"  Output scale applied: {scale}", flush=True)
    print(f"  Saved: {transfer_path}", flush=True)

# ====================================================================
# PPO fine-tuning
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"[INFO] Starting PPO fine-tuning", flush=True)
print(f"[INFO] max_iterations = {args_cli.max_iterations}", flush=True)
print(f"[INFO] action_penalty_weight = {args_cli.action_penalty_weight}", flush=True)
print(f"[INFO] transfer_output_scale = {args_cli.transfer_output_scale}", flush=True)
print(f"[INFO] num_envs = {B}", flush=True)
print(f"[INFO] log_dir = {log_dir}", flush=True)
print(f"{'='*60}", flush=True)

try:
    with torch.inference_mode(False):
        runner.learn(num_learning_iterations=int(args_cli.max_iterations), init_at_random_ep_len=True)
    print("\n[INFO] PPO fine-tuning complete", flush=True)
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
