#!/usr/bin/env python3
"""Diagnose why PPO cannot match greedy mapper."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose PPO vs greedy actions.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage0_yz_env_wrapper import Stage0YZEnvWrapper
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg
from rsl_rl.runners import OnPolicyRunner

cfg = MulagUpperLevelEnvCfg()
env = Stage0YZEnvWrapper(cfg, horizon_steps=10, action_scale_m=0.05, device=args_cli.device)
B = env.num_envs

# Load PPO
agent_cfg = MulagUpperLevelPPORunnerCfg()
agent_cfg.device = args_cli.device
agent_cfg.actor.hidden_dims = [64, 32]
agent_cfg.critic.hidden_dims = [64, 32]
runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=args_cli.device)
runner.load(args_cli.checkpoint)
ppo_model = runner.alg.actor_critic
print(f"Loaded: {args_cli.checkpoint}", flush=True)

num_eps = int(args_cli.num_episodes)
results = []

def greedy_action():
    ee = env._tcp_yz()
    err = env._target_yz - ee
    return torch.clamp(err / env._action_scale, -1.0, 1.0)

def ppo_action():
    obs = env._get_obs()
    td = env._to_tensordict(obs)
    with torch.no_grad():
        act = ppo_model.act_inference(td)["policy"]
    return act

def run_one_episode(action_fn):
    env.reset()
    ee0 = env._tcp_yz()[0].detach().cpu().numpy()
    target = env._target_yz[0].detach().cpu().numpy()
    target_delta = target - ee0

    actions = []; distances = []
    for step in range(20):
        act = action_fn()
        obs, reward, dones, extras = env.step(act)
        d = extras.get("log", {}).get("distance", torch.tensor(0.0))
        distances.append(float(d.mean().item() if d.numel() > 1 else d.item()))
        actions.append(act[0].detach().cpu().numpy())
        if dones[0].item():
            break

    actions = np.array(actions)
    return {
        "target_delta_yz": target_delta,
        "actions": actions,
        "action_mean": np.mean(actions, axis=0) if len(actions) > 0 else np.zeros(2),
        "action_std": np.std(actions, axis=0) if len(actions) > 0 else np.zeros(2),
        "action_norm_mean": float(np.mean(np.linalg.norm(actions, axis=1))),
        "distances": distances,
        "final_distance": distances[-1] if distances else 999,
        "steps": len(distances),
    }

print("=" * 60, flush=True)
print("PPO vs GREEDY ACTION DIAGNOSIS", flush=True)
print("=" * 60, flush=True)

# Collect paired data: same reset state, compare greedy vs PPO
all_greedy = []; all_ppo = []; all_zero = []

for ep in range(num_eps):
    # Seed the same reset for paired comparison
    np.random.seed(ep * 3)
    torch.manual_seed(ep * 3)

    env.reset()
    target_delta = env._target_yz[0].detach().cpu().numpy() - env._tcp_yz()[0].detach().cpu().numpy()

    # Greedy action (just compute, don't execute)
    g_act = greedy_action()[0].detach().cpu().numpy()

    # PPO action (just compute, don't execute)
    p_act = ppo_action()[0].detach().cpu().numpy()

    # Action stats
    action_mse = float(np.mean((g_act - p_act)**2))
    g_norm = float(np.linalg.norm(g_act))
    p_norm = float(np.linalg.norm(p_act))
    dot = float(np.dot(g_act, p_act))
    cosine = dot / max(g_norm * p_norm, 1e-9) if g_norm > 1e-6 and p_norm > 1e-6 else 0.0

    # Now run zero, greedy, PPO episodes from same reset
    # Zero
    env.reset()
    for _ in range(20):
        obs, reward, dones, extras = env.step(torch.zeros((B, 2), device=args_cli.device))
        if dones[0].item(): break
    z_dist = float(extras.get("log", {}).get("distance", torch.tensor(0.0)).mean().item())

    # Greedy
    env.reset()
    for _ in range(20):
        obs, reward, dones, extras = env.step(greedy_action())
        if dones[0].item(): break
    g_dist = float(extras.get("log", {}).get("distance", torch.tensor(0.0)).mean().item())

    # PPO
    env.reset()
    for _ in range(20):
        obs, reward, dones, extras = env.step(ppo_action())
        if dones[0].item(): break
    p_dist = float(extras.get("log", {}).get("distance", torch.tensor(0.0)).mean().item())

    all_greedy.append({"dist": g_dist, "act_norm": g_norm, "act": g_act})
    all_ppo.append({"dist": p_dist, "act_norm": p_norm, "act": p_act})
    all_zero.append(z_dist)

    if ep < 5:
        print(f"  ep{ep}: target_delta=[{target_delta[0]:+.3f},{target_delta[1]:+.3f}] "
              f"g_act=[{g_act[0]:+.3f},{g_act[1]:+.3f}] p_act=[{p_act[0]:+.3f},{p_act[1]:+.3f}] "
              f"cos={cosine:+.2f} mse={action_mse:.4f} "
              f"z_dist={z_dist:.3f} g_dist={g_dist:.3f} p_dist={p_dist:.3f}", flush=True)

# Summary
n = len(all_greedy)
mean_z = np.mean(all_zero)
mean_g = np.mean([r["dist"] for r in all_greedy])
mean_p = np.mean([r["dist"] for r in all_ppo])
mean_g_norm = np.mean([r["act_norm"] for r in all_greedy])
mean_p_norm = np.mean([r["act_norm"] for r in all_ppo])

# Action cosine across all paired states
cosines = []
for i in range(n):
    g = all_greedy[i]["act"]; p = all_ppo[i]["act"]
    gn = float(np.linalg.norm(g)); pn = float(np.linalg.norm(p))
    if gn > 1e-6 and pn > 1e-6:
        cosines.append(float(np.dot(g, p) / (gn * pn)))

mean_cos = np.mean(cosines) if cosines else 0.0

print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Episodes: {n}", flush=True)
print(f"  Zero mean dist:     {mean_z:.3f} m", flush=True)
print(f"  Greedy mean dist:   {mean_g:.3f} m", flush=True)
print(f"  PPO mean dist:      {mean_p:.3f} m", flush=True)
print(f"  Greedy action norm: {mean_g_norm:.3f}", flush=True)
print(f"  PPO action norm:    {mean_p_norm:.3f}", flush=True)
print(f"  Action cosine:      {mean_cos:.3f}", flush=True)

print(f"\n  Diagnosis:", flush=True)
if mean_p_norm < 0.01:
    print(f"    PPO COLLAPSED TO NO-OP: action_norm={mean_p_norm:.4f} → increase progress reward, reduce action penalty", flush=True)
elif mean_cos < -0.5:
    print(f"    PPO ACTION SIGN WRONG: cosine={mean_cos:.3f} → fix mapper/action sign", flush=True)
elif mean_cos > 0.8:
    print(f"    PPO LEARNED DIRECTION, MAGNITUDE ISSUE: cosine={mean_cos:.3f} → tune action_scale, reward weights", flush=True)
elif abs(mean_cos) < 0.3:
    print(f"    PPO NOT LEARNING MAPPING: cosine={mean_cos:.3f} → BC pretraining needed", flush=True)
elif mean_p_norm > 2.0 * mean_g_norm:
    print(f"    PPO TOO NOISY: ppo_norm={mean_p_norm:.3f} vs greedy_norm={mean_g_norm:.3f} → reduce action std/scale", flush=True)
else:
    print(f"    MIXED: cosine={mean_cos:.3f} ppo_norm={mean_p_norm:.3f} → review detailed logs", flush=True)

env.close()
simulation_app.close()
