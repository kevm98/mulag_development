#!/usr/bin/env python3
"""PPO Stage-0 Y-Z warm-started from BC checkpoint — corrected no-pin dynamics.

Gates:
  A: zero-action no-pin hold (mean_jump < 0.05, max_jump < 0.10)  [HARD FAIL]
  B: BC->PPO weight copy cosine > 0.8                             [HARD FAIL]
  C: BC-init rollout sanity (warning only — old BC is stale)       [WARN ONLY]
"""
from __future__ import annotations
import argparse, os, sys, subprocess
from datetime import datetime
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage-0 Y-Z PPO warm-start from BC (no-pin).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=300)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--action_scale_m", type=float, default=0.05)
parser.add_argument("--num_steps_per_env", type=int, default=8)
parser.add_argument("--save_interval", type=int, default=25)
parser.add_argument("--bc_checkpoint", type=str, default="outputs/bc_actor_yz.pt")
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--force_rate_limit_fraction", type=float, default=0.01)
parser.add_argument("--velocity_filter_beta", type=float, default=0.8)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.005)
parser.add_argument("--experiment_name", type=str, default="stage0_yz_nopin_bc")
parser.add_argument("--eval_after_train", action="store_true", default=False)
parser.add_argument("--eval_episodes", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage0_yz_env_wrapper import Stage0YZEnvWrapper
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg
from rsl_rl.runners import OnPolicyRunner

# --- Print config ---
print(f"[INFO] corrected no-pin Stage-0 config", flush=True)
print(f"[INFO] obs_dim = 4", flush=True)
print(f"[INFO] act_dim = 2", flush=True)
print(f"[INFO] horizon_steps = {args_cli.horizon_steps}", flush=True)
print(f"[INFO] action_scale = {args_cli.action_scale_m}", flush=True)
print(f"[INFO] kp_scale = {args_cli.kp_scale}", flush=True)
print(f"[INFO] kd_scale = {args_cli.kd_scale}", flush=True)
print(f"[INFO] force_rate_limit_fraction = {args_cli.force_rate_limit_fraction}", flush=True)
print(f"[INFO] velocity_filter_beta = {args_cli.velocity_filter_beta}", flush=True)
print(f"[INFO] max_cylinder_delta_m = {args_cli.max_cylinder_delta_m}", flush=True)
print(f"[INFO] _pin_root call sites disabled", flush=True)
print(f"[INFO] BC checkpoint = {args_cli.bc_checkpoint}", flush=True)

# --- Build env with corrected gains ---
cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
env_wrapper = Stage0YZEnvWrapper(cfg, horizon_steps=int(args_cli.horizon_steps),
                                 action_scale_m=float(args_cli.action_scale_m),
                                 device=args_cli.device,
                                 kp_scale=float(args_cli.kp_scale),
                                 kd_scale=float(args_cli.kd_scale),
                                 force_rate_limit=float(args_cli.force_rate_limit_fraction),
                                 velocity_filter_beta=float(args_cli.velocity_filter_beta))
B = env_wrapper.num_envs
device = args_cli.device

log_root = os.path.join("logs", "rsl_rl", args_cli.experiment_name)
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(log_root, run_name); os.makedirs(log_dir, exist_ok=True)
print(f"[INFO] log_dir = {log_dir}", flush=True)

# ====================================================================
# GATE A: zero-action no-pin hold  [HARD FAIL]
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"GATE A: zero-action no-pin hold diagnostic", flush=True)
print(f"{'='*60}", flush=True)
env_wrapper.reset()
for _ in range(10):
    env_wrapper.step(torch.zeros((B, 2), device=device))
env_wrapper.resample_local_targets()

jumps = []; dists = []
GATE_A_STEPS = 20
for _ in range(GATE_A_STEPS):
    obs, reward, done, extras = env_wrapper.step(torch.zeros((B, 2), device=device))
    jumps.append(extras["log"]["ee_jump"].item())
    dists.append(extras["log"]["distance"].item())

mean_j = float(np.mean(jumps)); max_j = float(np.max(jumps))
mean_d = float(np.mean(dists))
print(f"  mean_jump = {mean_j:.4f} m  (gate < 0.05)", flush=True)
print(f"  max_jump  = {max_j:.4f} m  (gate < 0.10)", flush=True)
print(f"  mean_dist = {mean_d:.4f} m", flush=True)
gate_a_pass = (mean_j < 0.05 and max_j < 0.10)
print(f"  GATE A: {'PASS' if gate_a_pass else 'FAIL'}", flush=True)
if not gate_a_pass:
    print(f"[ABORT] Gate A failed — zero-action hold unstable. Check gains.", flush=True)
    env_wrapper.close()
    simulation_app.close()
    sys.exit(1)

# --- Build runner ---
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

runner = OnPolicyRunner(env_wrapper, agent_cfg.to_dict(), log_dir=log_dir, device=device)
ppo_actor = runner.alg.actor

# ====================================================================
# GATE B: BC->PPO weight copy + cosine check  [HARD FAIL]
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"GATE B: BC->PPO weight copy sanity", flush=True)
print(f"{'='*60}", flush=True)
bc_path = Path(args_cli.bc_checkpoint)
bc_ckpt = torch.load(str(bc_path), map_location='cpu', weights_only=False)
bc_state = bc_ckpt["actor_state_dict"]

ppo_sd = ppo_actor.state_dict()
copied = 0
for key, val in bc_state.items():
    if key in ppo_sd and ppo_sd[key].shape == val.shape:
        ppo_sd[key] = val.to(ppo_sd[key].device)
        copied += 1
ppo_actor.load_state_dict(ppo_sd, strict=False)
print(f"  Copied {copied} BC params into PPO actor", flush=True)

# Cosine check
dev = ppo_actor.mlp[0].weight.device
cosines = []
for key in ["mlp.0.weight", "mlp.2.weight", "mlp.4.weight"]:
    bw = bc_state[key].flatten().float().to(dev)
    pw = ppo_actor.state_dict()[key].flatten().float()
    c = torch.nn.functional.cosine_similarity(bw, pw, dim=0).item()
    cosines.append(c)
mean_cos = float(np.mean(cosines))
print(f"  Cosine mlp.0/2/4: {cosines[0]:.4f} {cosines[1]:.4f} {cosines[2]:.4f}  mean={mean_cos:.4f}", flush=True)
gate_b_pass = mean_cos > 0.8
print(f"  GATE B: {'PASS' if gate_b_pass else 'FAIL'}  (cosine > 0.8)", flush=True)
if not gate_b_pass:
    print(f"[ABORT] Gate B failed — BC weight copy mismatch.", flush=True)
    env_wrapper.close()
    simulation_app.close()
    sys.exit(1)

# --- Save BC-init checkpoint before any PPO updates ---
# --- Save BC-init checkpoint before any PPO updates ---
bc_init_path = os.path.join(log_dir, "model_bc_init.pt")
saved_dict = runner.alg.save()
saved_dict["iter"] = runner.current_learning_iteration
saved_dict["infos"] = {}
torch.save(saved_dict, bc_init_path)
print(f"\n[INFO] Saved bc_init checkpoint: {bc_init_path}", flush=True)

# ====================================================================
# GATE C: BC-init rollout sanity  [WARNING ONLY]
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"GATE C: BC-init rollout vs baselines (warning only)", flush=True)
print(f"{'='*60}", flush=True)

def evaluate_baseline(action_fn, n_eps=10):
    env_wrapper.reset()
    for _ in range(5):
        env_wrapper.step(torch.zeros((B, 2), device=device))
    env_wrapper.resample_local_targets()
    ep_dists = []
    for ep in range(n_eps):
        ep_dist = []
        for _ in range(env_wrapper.max_episode_length):
            action = action_fn()
            obs, reward, done, extras = env_wrapper.step(action)
            ep_dist.append(extras["log"]["distance"].item())
            if done.any():
                break
        ep_dists.append(float(np.mean(ep_dist)) if ep_dist else 0.0)
    return float(np.mean(ep_dists)) if ep_dists else float('inf')

z_dist = evaluate_baseline(lambda: torch.zeros((B, 2), device=device))
print(f"  zero   mean_dist = {z_dist:.4f} m", flush=True)

r_dist = evaluate_baseline(lambda: 2.0 * torch.rand((B, 2), device=device) - 1.0)
print(f"  random mean_dist = {r_dist:.4f} m", flush=True)

def bc_fn():
    obs = env_wrapper._get_obs()
    td = env_wrapper._to_tensordict(obs)
    with torch.no_grad():
        actions = ppo_actor(td)
    return actions
bc_dist = evaluate_baseline(bc_fn)
print(f"  bc_init mean_dist = {bc_dist:.4f} m", flush=True)

if bc_dist < z_dist and bc_dist < r_dist:
    print(f"  GATE C: PASS  (bc_init beats both zero and random)", flush=True)
elif bc_dist < z_dist:
    print(f"  GATE C: WARN  (bc_init beats zero but not random; old BC prior may be stale)", flush=True)
else:
    print(f"  GATE C: WARN  (bc_init worse than zero; old BC prior is stale, PPO may still recover)", flush=True)

# ====================================================================
# PPO fine-tuning
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"[INFO] Starting PPO fine-tuning", flush=True)
print(f"[INFO] max_iterations = {args_cli.max_iterations}", flush=True)
print(f"[INFO] num_envs = {B}", flush=True)
print(f"[INFO] num_steps_per_env = {args_cli.num_steps_per_env}", flush=True)
print(f"[INFO] save_interval = {args_cli.save_interval}", flush=True)
print(f"[INFO] log_dir = {log_dir}", flush=True)
print(f"[INFO] lr=3e-4, entropy=0.001, obs_norm=OFF, no-pin", flush=True)
print(f"[INFO] kp_scale={args_cli.kp_scale}, kd_scale={args_cli.kd_scale}", flush=True)
print(f"{'='*60}", flush=True)

try:
    with torch.inference_mode(False):
        runner.learn(num_learning_iterations=int(args_cli.max_iterations), init_at_random_ep_len=True)
    print("\n[INFO] PPO fine-tuning complete", flush=True)
except Exception as e:
    print(f"\n[ERROR] PPO fine-tuning failed: {e}", flush=True)
    import traceback; traceback.print_exc()

# --- Save final checkpoint ---
final_path = os.path.join(log_dir, "model_final.pt")
try:
    saved_dict = runner.alg.save()
    saved_dict["iter"] = runner.current_learning_iteration
    saved_dict["infos"] = {}
    torch.save(saved_dict, final_path)
    print(f"\n[INFO] Saved final checkpoint: {final_path}", flush=True)
except Exception as e:
    print(f"[WARN] Could not save final checkpoint: {e}", flush=True)

# Find latest checkpoint
ckpt_files = sorted(Path(log_dir).glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
latest_ckpt = str(ckpt_files[-1]) if ckpt_files else None
if latest_ckpt:
    print(f"[INFO] Latest checkpoint = {latest_ckpt}", flush=True)

# ====================================================================
# Auto-evaluation after training
# ====================================================================
if args_cli.eval_after_train and latest_ckpt:
    print(f"\n{'='*60}", flush=True)
    print(f"[INFO] Running auto-evaluation on latest checkpoint", flush=True)
    print(f"{'='*60}", flush=True)

    eval_cmd = [
        sys.executable,
        str(ROOT / "kevin_integration/new_observations_pipeline/scripts/evaluate_stage0_yz_policy.py"),
        "--checkpoint", latest_ckpt,
        "--num_envs", "1",
        "--num_episodes", str(args_cli.eval_episodes),
        "--settle_steps", "30",
        "--kp_scale", str(args_cli.kp_scale),
        "--kd_scale", str(args_cli.kd_scale),
        "--force_rate_limit_fraction", str(args_cli.force_rate_limit_fraction),
        "--velocity_filter_beta", str(args_cli.velocity_filter_beta),
        "--headless",
        "--device", args_cli.device,
    ]
    print(f"[INFO] eval cmd: {' '.join(eval_cmd)}", flush=True)

    try:
        result = subprocess.run(eval_cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"[WARN] Evaluation returned exit code {result.returncode}", flush=True)
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", flush=True)

elif args_cli.eval_after_train:
    print(f"\n[WARN] --eval_after_train set but no checkpoint found in {log_dir}", flush=True)

env_wrapper.close()
simulation_app.close()
print("\nDone.", flush=True)
