#!/usr/bin/env python3
"""Evaluate Stage-0 Y-Z policy against fixed baselines."""
from __future__ import annotations
import argparse, sys, os, csv
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Stage-0 Y-Z policy.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output_csv", type=str, default=None)
parser.add_argument("--settle_steps", type=int, default=30)
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--force_rate_limit_fraction", type=float, default=0.01)
parser.add_argument("--velocity_filter_beta", type=float, default=0.8)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.005)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage0_yz_env_wrapper import Stage0YZEnvWrapper

# --- Build environment ---
cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
print(f"[INFO] Evaluation num_envs = {args_cli.num_envs}", flush=True)
env = Stage0YZEnvWrapper(cfg, horizon_steps=10, action_scale_m=0.05, device=args_cli.device,
                            kp_scale=float(args_cli.kp_scale),
                            kd_scale=float(args_cli.kd_scale),
                            force_rate_limit=float(args_cli.force_rate_limit_fraction),
                            velocity_filter_beta=float(args_cli.velocity_filter_beta))
B = env.num_envs
print(f"[INFO] Created env num_envs = {B}", flush=True)
print(f"[INFO] kp_scale = {args_cli.kp_scale}, kd_scale = {args_cli.kd_scale}, no-pin", flush=True)

# Load PPO policy if checkpoint provided
ppo_model = None
if args_cli.checkpoint and Path(args_cli.checkpoint).exists():
    from rsl_rl.runners import OnPolicyRunner
    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg

    # --- Inspect checkpoint metadata ---
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["actor_state_dict"]
    has_obs_normalizer = any(k.startswith("obs_normalizer") for k in sd.keys())
    obs_dim = sd["mlp.0.weight"].shape[1]
    act_dim = sd["mlp.4.weight"].shape[0]
    print(f"[INFO] checkpoint actor keys include obs_normalizer: {has_obs_normalizer}", flush=True)
    print(f"[INFO] checkpoint obs_dim: {obs_dim}", flush=True)
    print(f"[INFO] checkpoint act_dim: {act_dim}", flush=True)
    assert obs_dim == 4, f"Expected obs_dim=4, got {obs_dim}"
    assert act_dim == 2, f"Expected act_dim=2, got {act_dim}"

    # --- Build runner with matching architecture ---
    agent_cfg = MulagUpperLevelPPORunnerCfg()
    agent_cfg.device = args_cli.device
    agent_cfg.actor.hidden_dims = [64, 32]
    agent_cfg.critic.hidden_dims = [64, 32]
    # Match training: disable obs normalization if checkpoint has no normalizer
    if not has_obs_normalizer:
        agent_cfg.actor.obs_normalization = False
        agent_cfg.critic.obs_normalization = False
        print("[INFO] Disabled obs_normalization (checkpoint trained without it)", flush=True)
    else:
        print("[INFO] Keeping obs_normalization enabled (checkpoint has normalizer)", flush=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=args_cli.device)
    # Load manually to handle checkpoints without 'infos' key
    loaded_dict = torch.load(args_cli.checkpoint, map_location=args_cli.device, weights_only=False)
    runner.alg.load(loaded_dict, load_cfg=None, strict=True)
    if "iter" in loaded_dict:
        runner.current_learning_iteration = loaded_dict["iter"]
    ppo_model = runner.alg.actor
    print(f"Loaded checkpoint: {args_cli.checkpoint}", flush=True)

def settle_and_resample():
    """Run settle_steps zero-action steps (prevent truncation), then resample."""
    saved_max = env.max_episode_length
    env.max_episode_length = 999999
    for _ in range(settle_steps):
        env.step(zero_action)
    env.max_episode_length = saved_max
    env._ep_length.zero_()
    env.resample_local_targets()

def evaluate_policy(action_fn, label, num_eps):
    # --- Initial reset + settle + resample ---
    env.reset()
    settle_and_resample()

    # --- Debug: check state after reset ---
    ee = env._tcp_yz()
    target = env._target_yz
    target_err = torch.norm(target - ee, dim=-1).mean().item()
    print(f"[DEBUG] {label}: after reset+settle  ee_yz=({ee[0,0].item():.4f},{ee[0,1].item():.4f})  "
          f"target_yz=({target[0,0].item():.4f},{target[0,1].item():.4f})  "
          f"target_err={target_err:.4f}m  prev_action=({env._prev_action[0,0].item():.2f},{env._prev_action[0,1].item():.2f})",
          flush=True)

    distances = []; rewards_list = []; jumps = []; actions_list = []
    successes_10 = []; successes_05 = []; successes_03 = []
    episodes = 0
    episode_distances = []; episode_jumps = []; episode_actions = []; episode_rewards = []
    while episodes < num_eps:
        action = action_fn()
        obs, reward, done, extras = env.step(action)

        # --- Collect per-step stats ---
        d = extras.get("log", {}).get("distance", torch.tensor(0.0))
        d = float(d.mean().item() if d.numel() > 1 else d.item())
        episode_distances.append(d)
        aj = extras.get("log", {}).get("ee_jump", torch.tensor(0.0))
        aj_val = float(aj.mean().item() if aj.numel() > 1 else aj.item())
        episode_jumps.append(aj_val)

        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.ndim == 0: done_ids = done_ids.unsqueeze(0)
            for _ in done_ids:
                if episodes >= num_eps: break
                # Record episode-level aggregates
                distances.append(float(np.mean(episode_distances)) if episode_distances else 0.0)
                jumps.append(float(np.mean(episode_jumps)) if episode_jumps else 0.0)
                successes_10.append(1 if (episode_distances and episode_distances[-1] < 0.10) else 0)
                successes_05.append(1 if (episode_distances and episode_distances[-1] < 0.05) else 0)
                successes_03.append(1 if (episode_distances and episode_distances[-1] < 0.03) else 0)
                actions_list.append(float(np.mean(episode_actions)) if episode_actions else 0.0)
                rewards_list.append(float(np.mean(episode_rewards)) if episode_rewards else 0.0)
                episodes += 1
            # After episode termination, settle + resample before next episode
            if episodes < num_eps:
                settle_and_resample()
            episode_distances = []
            episode_jumps = []
            episode_actions = []
            episode_rewards = []
        else:
            an = extras.get("log", {}).get("action_norm", torch.tensor(0.0))
            episode_actions.append(float(an.mean().item() if an.numel() > 1 else an.item()))
            rw = extras.get("log", {}).get("reward_mean", torch.tensor(0.0))
            episode_rewards.append(float(rw.mean().item() if rw.numel() > 1 else rw.item()))
    if not distances: return {}
    return {
        "label": label,
        "episodes": len(distances),
        "mean_dist": float(np.mean(distances)),
        "median_dist": float(np.median(distances)),
        "std_dist": float(np.std(distances)),
        "sr_010": float(np.mean(successes_10)),
        "sr_005": float(np.mean(successes_05)),
        "sr_003": float(np.mean(successes_03)),
        "mean_jump": float(np.mean(jumps)),
        "mean_action_norm": float(np.mean(actions_list)),
        "mean_reward": float(np.mean(rewards_list)),
    }

num_eps = int(args_cli.num_episodes)
settle_steps = int(args_cli.settle_steps)
results = []
zero_action = torch.zeros((B, 2), device=args_cli.device)

# Zero action
r_zero = evaluate_policy(lambda: zero_action, "zero", num_eps)
results.append(r_zero)

# Random action
r_rand = evaluate_policy(lambda: 2.0 * torch.rand((B, 2), device=args_cli.device) - 1.0, "random", num_eps)
results.append(r_rand)

# Greedy mapper
def greedy_fn():
    ee = env._tcp_yz()
    err = env._target_yz - ee
    return torch.clamp(err / env._action_scale, -1.0, 1.0)
r_greedy = evaluate_policy(greedy_fn, "greedy", num_eps)
results.append(r_greedy)

# PPO policy
if ppo_model is not None:
    def ppo_fn():
        obs = env._get_obs()
        td = env._to_tensordict(obs)
        with torch.no_grad():
            actions = ppo_model(td)
        return actions
    r_ppo = evaluate_policy(ppo_fn, "ppo", num_eps)
    results.append(r_ppo)

print("=" * 60, flush=True)
print("EVALUATION", flush=True)
print("=" * 60, flush=True)
for r in results:
    if not r: continue
    print(f"\n  {r['label']}:", flush=True)
    print(f"    distance: mean={r['mean_dist']:.3f}m median={r['median_dist']:.3f}m std={r['std_dist']:.3f}m", flush=True)
    print(f"    success: 0.10m={r['sr_010']:.2f} 0.05m={r['sr_005']:.2f} 0.03m={r['sr_003']:.2f}", flush=True)
    print(f"    mean_jump={r['mean_jump']:.3f}m action_norm={r['mean_action_norm']:.3f} reward={r['mean_reward']:.3f}", flush=True)

# Write CSV if requested
if args_cli.output_csv:
    csv_path = Path(args_cli.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label","episodes","mean_dist","median_dist","std_dist",
                  "sr_010","sr_005","sr_003","mean_jump","mean_action_norm","mean_reward"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header: writer.writeheader()
        for r in results:
            if r: writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nWrote to {csv_path}", flush=True)

env.close()
simulation_app.close()
