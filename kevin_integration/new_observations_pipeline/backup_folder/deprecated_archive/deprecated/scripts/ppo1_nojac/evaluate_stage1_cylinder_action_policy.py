#!/usr/bin/env python3
"""Evaluate no-Jacobian cylinder-action PPO policy against baselines."""
from __future__ import annotations
import argparse, sys, os, time, json
from pathlib import Path
import numpy as np

print("[EVAL_NJ] launching app", flush=True)
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--workspace_json", type=str, default=None)
parser.add_argument("--trajectory_mode", type=str, default="sequence_easy")
parser.add_argument("--target_hold_steps", type=int, default=50)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--bias_calibration_time_s", type=float, default=5.0)
parser.add_argument("--post_reset_settle_steps", type=int, default=5)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.0025)
parser.add_argument("--action_smoothing_beta", type=float, default=0.8)
parser.add_argument("--action_rate_limit", type=float, default=0.1)
parser.add_argument("--controller", type=str, default="all",
                    choices=["all", "zero", "random", "ppo"])
parser.add_argument("--max_eval_steps_per_episode", type=int, default=200)
parser.add_argument("--eval_timeout_s", type=int, default=600)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[EVAL_NJ] importing env", flush=True)
import torch
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1_nojac.stage1_cylinder_action_env_wrapper import Stage1CylinderActionEnvWrapper

ws = {}
if args_cli.workspace_json and Path(args_cli.workspace_json).exists():
    with open(args_cli.workspace_json) as f:
        ws = json.load(f)
    print(f"[EVAL_NJ] workspace loaded", flush=True)

print("[EVAL_NJ] building env", flush=True)
cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
env = Stage1CylinderActionEnvWrapper(cfg,
    horizon_steps=10, device=args_cli.device,
    kp_scale=float(args_cli.kp_scale), kd_scale=float(args_cli.kd_scale),
    trajectory_mode=args_cli.trajectory_mode,
    target_hold_steps=int(args_cli.target_hold_steps),
    post_reset_settle_steps=int(args_cli.post_reset_settle_steps),
    backend=str(args_cli.backend),
    bias_calibration_time_s=float(args_cli.bias_calibration_time_s),
    max_cylinder_delta_m=float(args_cli.max_cylinder_delta_m),
    action_smoothing_beta=float(args_cli.action_smoothing_beta),
    action_rate_limit=float(args_cli.action_rate_limit),
    workspace_yz=ws)

B = env.num_envs
device = args_cli.device
print(f"[EVAL_NJ] env ready num_envs={B}", flush=True)

# Load checkpoint
ppo_model = None
if args_cli.checkpoint:
    cp = Path(args_cli.checkpoint)
    if not cp.exists():
        print(f"[EVAL_NJ] ERROR: checkpoint not found", flush=True)
        env.close(); simulation_app.close(); sys.exit(1)
    print(f"[EVAL_NJ] loading checkpoint", flush=True)
    try:
        from rsl_rl.runners import OnPolicyRunner
        from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg
        ckpt = torch.load(str(cp), map_location="cpu", weights_only=False)
        sd = ckpt.get("actor_state_dict") or ckpt.get("model_state_dict")
        if sd is None:
            print(f"[EVAL_NJ] ERROR: no actor_state_dict. Keys: {list(ckpt.keys())[:5]}", flush=True)
            env.close(); simulation_app.close(); sys.exit(1)
        agent_cfg = MulagUpperLevelPPORunnerCfg()
        agent_cfg.device = device
        agent_cfg.actor.hidden_dims = [128, 64]; agent_cfg.critic.hidden_dims = [128, 64]
        agent_cfg.actor.obs_normalization = False; agent_cfg.critic.obs_normalization = False
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=device)
        loaded = torch.load(str(cp), map_location=device, weights_only=False)
        runner.alg.load(loaded, load_cfg=None, strict=True)
        ppo_model = runner.alg.actor
        print(f"[EVAL_NJ] checkpoint loaded", flush=True)
    except Exception as e:
        print(f"[EVAL_NJ] ERROR: {e}", flush=True)
        env.close(); simulation_app.close(); sys.exit(1)

zero_action = torch.zeros((B, 4), device=device)
max_eps = int(args_cli.max_eval_steps_per_episode)

def settle():
    for _ in range(30):
        env.step(zero_action)
    env._ep_length.zero_()
    ee = env._tcp_yz()
    for i in range(B):
        env._traj_params[i] = env._init_trajectory(ee[i:i+1], args_cli.trajectory_mode)
        if 'sequence' in args_cli.trajectory_mode:
            yr = env._target_y_range; zr = env._target_z_range
            dy = (torch.rand(1, device=device) * 2 * yr - yr)
            dz = (torch.rand(1, device=device) * 2 * zr - zr)
            env._target_yz[i] = ee[i] + torch.cat([dy, dz], dim=-1)
            env._target_vel_yz[i].zero_()
        else:
            env._step_trajectory(i)
    env._prev_action.zero_()
    env._processed_action.zero_()
    env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)

def evaluate_policy(action_fn, label, num_eps):
    print(f"[EVAL_NJ] {label} ({num_eps} eps)", flush=True)
    env.reset(); settle()
    dists, jumps, s10, s05, s03, anorms = [], [], [], [], [], []
    ep, step_in_ep = 0, 0
    pd, pj, ps10, ps05, ps03, pan = [], [], [], [], [], []
    t0 = time.time()

    while ep < num_eps:
        if time.time() - t0 > args_cli.eval_timeout_s:
            print(f"  TIMEOUT after {ep} eps", flush=True); break
        a = action_fn()
        obs, reward, done, extras = env.step(a)
        d = extras["log"]["distance"].item()
        pd.append(d); pj.append(extras["log"]["ee_jump"].item())
        pan.append(extras["log"]["action_norm"].item())
        ps10.append(1.0 if d < 0.10 else 0.0)
        ps05.append(1.0 if d < 0.05 else 0.0)
        ps03.append(1.0 if d < 0.03 else 0.0)
        step_in_ep += 1

        if done.any() or step_in_ep >= max_eps:
            for _ in range(min(B, 1)):
                if ep >= num_eps: break
                if pd:
                    dists.append(float(np.mean(pd))); jumps.append(float(np.mean(pj)))
                    s10.append(float(np.mean(ps10))); s05.append(float(np.mean(ps05)))
                    s03.append(float(np.mean(ps03))); anorms.append(float(np.mean(pan)))
                ep += 1
                if ep % 10 == 0:
                    print(f"  {label}: {ep}/{num_eps} eps dist={np.mean(dists[-10:]):.4f} sr10={np.mean(s10[-10:]):.3f} t={time.time()-t0:.0f}s", flush=True)
            if ep < num_eps: settle()
            pd, pj, ps10, ps05, ps03, pan = [], [], [], [], [], []
            step_in_ep = 0

    if not dists: return {}
    return {"label": label,
            "mean_dist": float(np.mean(dists)), "median_dist": float(np.median(dists)),
            "sr10": float(np.mean(s10)), "sr05": float(np.mean(s05)), "sr03": float(np.mean(s03)),
            "mean_jump": float(np.mean(jumps)), "max_jump": float(np.max(jumps)),
            "action_norm": float(np.mean(anorms))}

num_eps = int(args_cli.num_episodes)
ctrl = args_cli.controller
results = []

if ctrl in ("all", "zero"):
    r = evaluate_policy(lambda: zero_action, "zero", num_eps); results.append(r)
if ctrl in ("all", "random"):
    r = evaluate_policy(lambda: torch.rand((B, 4), device=device) * 2 - 1, "random", num_eps); results.append(r)
if ctrl in ("all", "ppo") and ppo_model is not None:
    def ppo_fn():
        obs = env._get_obs(); td = env._to_tensordict(obs)
        with torch.no_grad(): return ppo_model(td)
    r = evaluate_policy(ppo_fn, "ppo", num_eps); results.append(r)

print("=" * 60, flush=True)
print("NO-JACOBIAN EVALUATION", flush=True)
print(f"mode={args_cli.trajectory_mode}", flush=True)
print("=" * 60, flush=True)
for r in results:
    if not r: continue
    print(f"\n  {r['label']}:", flush=True)
    print(f"    dist: mean={r['mean_dist']:.4f}m median={r['median_dist']:.4f}m", flush=True)
    print(f"    sr10={r['sr10']:.4f} sr05={r['sr05']:.4f} sr03={r['sr03']:.4f}", flush=True)
    print(f"    mean_jump={r['mean_jump']:.4f}m max_jump={r['max_jump']:.4f}m action_norm={r['action_norm']:.4f}", flush=True)

print("\n[EVAL_NJ] closing", flush=True)
env.close(); simulation_app.close()
print("[EVAL_NJ] done", flush=True)
