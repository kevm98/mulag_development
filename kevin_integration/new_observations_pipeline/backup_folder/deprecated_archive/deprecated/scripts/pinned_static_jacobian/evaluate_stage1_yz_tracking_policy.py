#!/usr/bin/env python3
"""Evaluate Stage-1 Y-Z tracking policy against baselines.
Fixed: proper launch order, phase prints, --controller flag, progress output.
"""
from __future__ import annotations
import argparse, sys, os, time
from pathlib import Path
import numpy as np

# === Phase 1: AppLauncher first (before any IsaacLab/torch imports) ===
print("[EVAL] launching app", flush=True)
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Evaluate Stage-1 Y-Z tracking policy.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--settle_steps", type=int, default=30)
parser.add_argument("--trajectory_mode", type=str, default="sequence")
parser.add_argument("--target_hold_steps", type=int, default=50)
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--force_rate_limit_fraction", type=float, default=0.01)
parser.add_argument("--velocity_filter_beta", type=float, default=0.8)
parser.add_argument("--target_y_range", type=float, default=0.03)
parser.add_argument("--target_z_range", type=float, default=0.01)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--bias_calibration_time_s", type=float, default=5.0)
parser.add_argument("--post_reset_settle_steps", type=int, default=5)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.005)
parser.add_argument("--reward_mode", type=str, default="default")
parser.add_argument("--action_penalty_weight", type=float, default=0.005)
parser.add_argument("--ee_jump_penalty_weight", type=float, default=0.05)
parser.add_argument("--action_smoothing_beta", type=float, default=0.0)
parser.add_argument("--action_rate_limit", type=float, default=-1.0)
parser.add_argument("--controller", type=str, default="all",
                    choices=["all", "zero", "random", "greedy", "ppo"],
                    help="Which controller(s) to evaluate.")
parser.add_argument("--max_eval_steps_per_episode", type=int, default=200)
parser.add_argument("--eval_timeout_s", type=int, default=600)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === Phase 2: Now import project modules ===
print("[EVAL] importing env", flush=True)
import torch
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage1_yz_tracking_env_wrapper import Stage1YZTrackingEnvWrapper

print("[EVAL] building env", flush=True)
cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
env = Stage1YZTrackingEnvWrapper(cfg, horizon_steps=10, action_scale_m=0.05, device=args_cli.device,
                                  kp_scale=float(args_cli.kp_scale), kd_scale=float(args_cli.kd_scale),
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
                                  action_rate_limit=float(args_cli.action_rate_limit))
B = env.num_envs
device = args_cli.device
env._target_y_range = float(args_cli.target_y_range)
env._target_z_range = float(args_cli.target_z_range)
print(f"[EVAL] env ready num_envs={B} mode={args_cli.trajectory_mode}", flush=True)

# === Phase 3: Load checkpoint if provided ===
ppo_model = None
if args_cli.checkpoint:
    print(f"[EVAL] loading checkpoint: {args_cli.checkpoint}", flush=True)
    cp = Path(args_cli.checkpoint)
    if not cp.exists():
        print(f"[EVAL] ERROR: checkpoint not found: {args_cli.checkpoint}", flush=True)
        env.close(); simulation_app.close(); sys.exit(1)
    try:
        from rsl_rl.runners import OnPolicyRunner
        from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg
        ckpt = torch.load(str(cp), map_location="cpu", weights_only=False)
        if "actor_state_dict" not in ckpt:
            if "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            else:
                print(f"[EVAL] ERROR: no actor_state_dict. Keys: {list(ckpt.keys())[:5]}", flush=True)
                env.close(); simulation_app.close(); sys.exit(1)
        else:
            sd = ckpt["actor_state_dict"]
        has_norm = any(k.startswith("obs_normalizer") for k in sd.keys())
        obs_dim = sd["mlp.0.weight"].shape[1]
        act_dim = sd["mlp.4.weight"].shape[0]
        print(f"[EVAL] ckpt obs_dim={obs_dim} act_dim={act_dim} has_norm={has_norm}", flush=True)
        agent_cfg = MulagUpperLevelPPORunnerCfg()
        agent_cfg.device = device
        agent_cfg.actor.hidden_dims = [64, 32]
        agent_cfg.critic.hidden_dims = [64, 32]
        if not has_norm:
            agent_cfg.actor.obs_normalization = False
            agent_cfg.critic.obs_normalization = False
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=device)
        loaded = torch.load(str(cp), map_location=device, weights_only=False)
        runner.alg.load(loaded, load_cfg=None, strict=True)
        if "iter" in loaded:
            runner.current_learning_iteration = loaded["iter"]
        ppo_model = runner.alg.actor
        print(f"[EVAL] checkpoint loaded", flush=True)
    except Exception as e:
        print(f"[EVAL] ERROR loading checkpoint: {e}", flush=True)
        import traceback; traceback.print_exc()
        env.close(); simulation_app.close(); sys.exit(1)

zero_action = torch.zeros((B, 2), device=device)
max_ep_steps = int(args_cli.max_eval_steps_per_episode)

def settle_and_resample():
    saved = env.max_episode_length
    env.max_episode_length = 999999
    for _ in range(int(args_cli.settle_steps)):
        env.step(zero_action)
    env.max_episode_length = saved
    env._ep_length.zero_()
    ee = env._tcp_yz()
    modes = ['sequence', 'line', 'circle', 'curve']
    for i in range(B):
        if args_cli.trajectory_mode == 'mixed':
            mode = modes[i % len(modes)]
        else:
            mode = args_cli.trajectory_mode
        env._traj_mode_per_env[i] = mode
        env._traj_params[i] = env._init_trajectory(ee[i:i+1], mode)
        env._step_trajectory(i)
    env._prev_action.zero_()
    env._processed_action.zero_()
    env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)

def evaluate_policy(action_fn, label, num_eps):
    print(f"[EVAL] starting {label} rollouts ({num_eps} eps)", flush=True)
    env.reset()
    settle_and_resample()
    ep_dists = []; ep_jumps = []; ep_anorms = []; ep_rewards = []
    successes_10 = []; successes_05 = []; successes_03 = []
    episodes = 0
    per_ep_dist = []; per_ep_jump = []; per_ep_an = []; per_ep_rew = []
    per_ep_s10 = []; per_ep_s05 = []; per_ep_s03 = []
    step_in_ep = 0
    t_start = time.time()

    while episodes < num_eps:
        if time.time() - t_start > args_cli.eval_timeout_s:
            print(f"[EVAL] {label}: TIMEOUT after {episodes} eps", flush=True)
            break
        action = action_fn()
        obs, reward, done, extras = env.step(action)
        d = extras["log"]["distance"].item()
        per_ep_dist.append(d)
        per_ep_jump.append(extras["log"]["ee_jump"].item())
        per_ep_an.append(extras["log"].get("processed_action_norm",
                          extras["log"].get("action_norm", torch.tensor(0.0))).item())
        per_ep_rew.append(extras["log"]["reward_mean"].item())
        per_ep_s10.append(1.0 if d < 0.10 else 0.0)
        per_ep_s05.append(1.0 if d < 0.05 else 0.0)
        per_ep_s03.append(1.0 if d < 0.03 else 0.0)
        step_in_ep += 1

        if done.any() or step_in_ep >= max_ep_steps:
            done_ids = done.nonzero(as_tuple=False).squeeze(-1) if done.any() else torch.arange(B, device=device)
            if done_ids.ndim == 0:
                done_ids = done_ids.unsqueeze(0)
            for _ in done_ids:
                if episodes >= num_eps:
                    break
                if per_ep_dist:
                    ep_dists.append(float(np.mean(per_ep_dist)))
                    ep_jumps.append(float(np.mean(per_ep_jump)))
                    ep_anorms.append(float(np.mean(per_ep_an)))
                    ep_rewards.append(float(np.mean(per_ep_rew)))
                    successes_10.append(float(np.mean(per_ep_s10)))
                    successes_05.append(float(np.mean(per_ep_s05)))
                    successes_03.append(float(np.mean(per_ep_s03)))
                episodes += 1
                if episodes % 10 == 0:
                    elapsed = time.time() - t_start
                    print(f"[EVAL] {label}: {episodes}/{num_eps} eps, "
                          f"dist={np.mean(ep_dists[-10:]):.4f}m, "
                          f"sr10={np.mean(successes_10[-10:]):.3f}, "
                          f"t={elapsed:.0f}s", flush=True)
            if episodes < num_eps:
                settle_and_resample()
            per_ep_dist = []; per_ep_jump = []; per_ep_an = []; per_ep_rew = []
            per_ep_s10 = []; per_ep_s05 = []; per_ep_s03 = []
            step_in_ep = 0

    elapsed = time.time() - t_start
    print(f"[EVAL] {label}: done {episodes} eps in {elapsed:.0f}s", flush=True)
    if not ep_dists:
        return {}
    return {"label": label, "episodes": len(ep_dists),
            "mean_dist": float(np.mean(ep_dists)),
            "median_dist": float(np.median(ep_dists)),
            "std_dist": float(np.std(ep_dists)),
            "sr_010": float(np.mean(successes_10)),
            "sr_005": float(np.mean(successes_05)),
            "sr_003": float(np.mean(successes_03)),
            "mean_jump": float(np.mean(ep_jumps)),
            "mean_action_norm": float(np.mean(ep_anorms)),
            "mean_reward": float(np.mean(ep_rewards))}

num_eps = int(args_cli.num_episodes)
ctrl = args_cli.controller
results = []

print(f"[EVAL] starting rollouts controller={ctrl}", flush=True)

if ctrl in ("all", "zero"):
    r = evaluate_policy(lambda: zero_action, "zero", num_eps); results.append(r)
if ctrl in ("all", "random"):
    r = evaluate_policy(lambda: 2.0 * torch.rand((B, 2), device=device) - 1.0, "random", num_eps); results.append(r)
if ctrl in ("all", "greedy"):
    def greedy_fn():
        ee = env._tcp_yz(); err = env._target_yz - ee
        return torch.clamp(err / env._action_scale, -1.0, 1.0)
    r = evaluate_policy(greedy_fn, "greedy", num_eps); results.append(r)
if ctrl in ("all", "ppo") and ppo_model is not None:
    def ppo_fn():
        obs = env._get_obs(); td = env._to_tensordict(obs)
        with torch.no_grad():
            actions = ppo_model(td)
        return actions
    r = evaluate_policy(ppo_fn, "ppo", num_eps); results.append(r)
elif ctrl == "ppo" and ppo_model is None:
    print("[EVAL] ERROR: --controller ppo but no --checkpoint given", flush=True)

print("=" * 60, flush=True)
print("STAGE-1 EVALUATION RESULTS", flush=True)
print(f"mode={args_cli.trajectory_mode} backend={args_cli.backend}", flush=True)
print(f"y_range={args_cli.target_y_range} z_range={args_cli.target_z_range} hold={args_cli.target_hold_steps}", flush=True)
print(f"num_envs={B} num_episodes={num_eps}", flush=True)
print("=" * 60, flush=True)
for r in results:
    if not r:
        continue
    print(f"\n  {r['label']}:", flush=True)
    print(f"    distance: mean={r['mean_dist']:.4f}m median={r['median_dist']:.4f}m std={r['std_dist']:.4f}m", flush=True)
    print(f"    success: 0.10m={r['sr_010']:.4f} 0.05m={r['sr_005']:.4f} 0.03m={r['sr_003']:.4f}", flush=True)
    print(f"    mean_jump={r['mean_jump']:.4f}m action_norm={r['mean_action_norm']:.4f} reward={r['mean_reward']:.4f}", flush=True)

print("\n[EVAL] closing", flush=True)
env.close()
simulation_app.close()
print("[EVAL] done", flush=True)
