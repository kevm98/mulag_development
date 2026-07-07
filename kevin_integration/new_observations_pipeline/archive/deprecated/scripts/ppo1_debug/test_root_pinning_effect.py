#!/usr/bin/env python3
"""A/B test: root pinning vs no-pin effect on tracking quality."""
from __future__ import annotations
import argparse, sys, os, time
from pathlib import Path
import numpy as np

print("[PIN_TEST] launching", flush=True)
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=30)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--bias_calibration_time_s", type=float, default=5.0)
parser.add_argument("--post_reset_settle_steps", type=int, default=5)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.005)
parser.add_argument("--action_smoothing_beta", type=float, default=0.8)
parser.add_argument("--action_rate_limit", type=float, default=0.1)
parser.add_argument("--target_y_range", type=float, default=0.03)
parser.add_argument("--target_z_range", type=float, default=0.01)
parser.add_argument("--target_hold_steps", type=int, default=50)
parser.add_argument("--eval_timeout_s", type=int, default=300)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage1_yz_tracking_env_wrapper import Stage1YZTrackingEnvWrapper

B = int(args_cli.num_envs)
device = args_cli.device
MAX_CYL_DELTA = float(args_cli.max_cylinder_delta_m)
ACTION_SCALE = 0.05

def build_env(pin_root):
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = B
    env = Stage1YZTrackingEnvWrapper(
        cfg, horizon_steps=10, action_scale_m=ACTION_SCALE, device=device,
        kp_scale=float(args_cli.kp_scale), kd_scale=float(args_cli.kd_scale),
        force_rate_limit=0.01, velocity_filter_beta=0.8,
        trajectory_mode="sequence",
        target_hold_steps=int(args_cli.target_hold_steps),
        post_reset_settle_steps=int(args_cli.post_reset_settle_steps),
        backend=args_cli.backend,
        bias_calibration_time_s=float(args_cli.bias_calibration_time_s),
        action_smoothing_beta=float(args_cli.action_smoothing_beta),
        action_rate_limit=float(args_cli.action_rate_limit),
        reward_mode="distance_only",
    )
    env._target_y_range = float(args_cli.target_y_range)
    env._target_z_range = float(args_cli.target_z_range)
    if not pin_root:
        # Disable pinning
        env._pin_root = lambda: None
    return env

zero_action = None  # set after env created

def evaluate_controller(env, label, num_eps, action_fn):
    print(f"\n[PIN_TEST] {label} ({num_eps} eps)", flush=True)
    env.reset()
    # Settle
    for _ in range(30):
        env.step(torch.zeros((B, 2), device=device))
    env._ep_length.zero_()
    ee = env._tcp_yz()
    for i in range(B):
        env._traj_params[i] = env._init_trajectory(ee[i:i+1], 'sequence')
        yr = env._target_y_range; zr = env._target_z_range
        dy = (torch.rand(1, device=device) * 2 * yr - yr)
        dz = (torch.rand(1, device=device) * 2 * zr - zr)
        env._target_yz[i] = ee[i] + torch.cat([dy, dz], dim=-1)
        env._target_vel_yz[i].zero_()
    env._prev_action.zero_()
    env._processed_action.zero_()
    env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)

    dists, jumps, successes_10, successes_05, successes_03 = [], [], [], [], []
    episodes = 0
    per_ep_d = []; per_ep_j = []; per_ep_s10 = []; per_ep_s05 = []; per_ep_s03 = []
    saved_max = env.max_episode_length
    env.max_episode_length = 200
    t0 = time.time()
    step_in_ep = 0

    while episodes < num_eps:
        if time.time() - t0 > args_cli.eval_timeout_s:
            print(f"  TIMEOUT after {episodes} eps", flush=True)
            break
        a = action_fn()
        obs, reward, done, extras = env.step(a)
        d = extras["log"]["distance"].item()
        per_ep_d.append(d)
        per_ep_j.append(extras["log"]["ee_jump"].item())
        per_ep_s10.append(1.0 if d < 0.10 else 0.0)
        per_ep_s05.append(1.0 if d < 0.05 else 0.0)
        per_ep_s03.append(1.0 if d < 0.03 else 0.0)
        step_in_ep += 1
        if done.any() or step_in_ep >= 200:
            for _ in (done.nonzero(as_tuple=False).squeeze(-1) if done.any() else torch.arange(min(B, 1), device=device)):
                if episodes >= num_eps: break
                if per_ep_d:
                    dists.append(float(np.mean(per_ep_d)))
                    jumps.append(float(np.mean(per_ep_j)))
                    successes_10.append(float(np.mean(per_ep_s10)))
                    successes_05.append(float(np.mean(per_ep_s05)))
                    successes_03.append(float(np.mean(per_ep_s03)))
                episodes += 1
            if episodes < num_eps:
                for _ in range(30):
                    env.step(torch.zeros((B, 2), device=device))
                env._ep_length.zero_()
                ee = env._tcp_yz()
                for i in range(B):
                    env._traj_params[i] = env._init_trajectory(ee[i:i+1], 'sequence')
                    yr = env._target_y_range; zr = env._target_z_range
                    dy = (torch.rand(1, device=device) * 2 * yr - yr)
                    dz = (torch.rand(1, device=device) * 2 * zr - zr)
                    env._target_yz[i] = ee[i] + torch.cat([dy, dz], dim=-1)
                    env._target_vel_yz[i].zero_()
                env._prev_action.zero_()
                env._processed_action.zero_()
                env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)
            per_ep_d = []; per_ep_j = []; per_ep_s10 = []; per_ep_s05 = []; per_ep_s03 = []
            step_in_ep = 0

    env.max_episode_length = saved_max
    if not dists:
        return {"label": label, "eps": 0, "mean_dist": 0, "sr10": 0, "sr05": 0, "sr03": 0, "mean_jump": 0, "max_jump": 0}
    return {
        "label": label, "eps": len(dists),
        "mean_dist": float(np.mean(dists)), "median_dist": float(np.median(dists)),
        "sr10": float(np.mean(successes_10)), "sr05": float(np.mean(successes_05)),
        "sr03": float(np.mean(successes_03)),
        "mean_jump": float(np.mean(jumps)), "max_jump": float(np.max(jumps)),
    }

num_eps = int(args_cli.num_episodes)
all_results = []

for mode, pin_root in [("A_no_pin", False), ("B_pin_root", True)]:
    print(f"\n{'='*60}", flush=True)
    print(f"[PIN_TEST] Mode: {mode}", flush=True)
    print(f"{'='*60}", flush=True)

    env = build_env(pin_root)
    zero_action_t = torch.zeros((B, 2), device=device)

    # Zero controller
    r = evaluate_controller(env, f"{mode}_zero", num_eps, lambda: zero_action_t)
    all_results.append(r)

    # Greedy controller
    def greedy_fn(e=env):
        err = e._target_yz - e._tcp_yz()
        return torch.clamp(err / ACTION_SCALE, -1.0, 1.0)
    r = evaluate_controller(env, f"{mode}_greedy", num_eps, greedy_fn)
    all_results.append(r)

    env.close()

# Print comparison
print(f"\n{'='*60}", flush=True)
print("ROOT PINNING A/B TEST RESULTS", flush=True)
print(f"{'='*60}", flush=True)
for r in all_results:
    print(f"\n  {r['label']} ({r['eps']} eps):", flush=True)
    print(f"    dist: mean={r['mean_dist']:.4f}m median={r['median_dist']:.4f}m", flush=True)
    print(f"    sr10={r['sr10']:.4f} sr05={r['sr05']:.4f} sr03={r['sr03']:.4f}", flush=True)
    print(f"    mean_jump={r['mean_jump']:.4f}m max_jump={r['max_jump']:.4f}m", flush=True)

# Key comparison
no_pin_zero = [r for r in all_results if "A_no_pin_zero" in r["label"]][0]
pin_zero = [r for r in all_results if "B_pin_root_zero" in r["label"]][0]
no_pin_greedy = [r for r in all_results if "A_no_pin_greedy" in r["label"]][0]
pin_greedy = [r for r in all_results if "B_pin_root_greedy" in r["label"]][0]

print(f"\n--- DECISION ---", flush=True)
print(f"zero: no_pin={no_pin_zero['mean_dist']:.4f} vs pin={pin_zero['mean_dist']:.4f} | max_jump: no_pin={no_pin_zero['max_jump']:.4f} vs pin={pin_zero['max_jump']:.4f}", flush=True)
print(f"greedy: no_pin={no_pin_greedy['mean_dist']:.4f} vs pin={pin_greedy['mean_dist']:.4f} | max_jump: no_pin={no_pin_greedy['max_jump']:.4f} vs pin={pin_greedy['max_jump']:.4f}", flush=True)

reject_pin = False
if pin_zero["max_jump"] > 0.08:
    reject_pin = True
    print(f"REJECT pin_root: max_jump={pin_zero['max_jump']:.4f} > 0.08", flush=True)
if pin_greedy["max_jump"] > 0.08:
    reject_pin = True
    print(f"REJECT pin_root: greedy max_jump={pin_greedy['max_jump']:.4f} > 0.08", flush=True)

if not reject_pin:
    print("pin_root ACCEPTED for now (both modes under jump threshold)", flush=True)
else:
    print("pin_root REJECTED — use no-pin mode", flush=True)

simulation_app.close()
print("\n[PIN_TEST] done", flush=True)
