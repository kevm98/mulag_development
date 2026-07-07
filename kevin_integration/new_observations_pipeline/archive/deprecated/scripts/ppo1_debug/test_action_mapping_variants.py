#!/usr/bin/env python3
"""Test action mapping variants WITHOUT PPO.
Compares different Jacobian/sign variants to find a working mapper.
"""
from __future__ import annotations
import argparse, sys, os, time
from pathlib import Path
import numpy as np

print("[MAPPER] launching app", flush=True)
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Mapper variant tests.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--trajectory_mode", type=str, default="sequence")
parser.add_argument("--target_hold_steps", type=int, default=50)
parser.add_argument("--target_y_range", type=float, default=0.03)
parser.add_argument("--target_z_range", type=float, default=0.01)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--bias_calibration_time_s", type=float, default=5.0)
parser.add_argument("--post_reset_settle_steps", type=int, default=5)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.005)
parser.add_argument("--num_rollouts", type=int, default=10)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--variants", type=str, default="all",
                    help="Comma-separated: A_original,B_flip_z_act,C_flip_z_row,D_flip_y_act,E_swap_yz,F_empirical,all")
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

# Original Jacobian
J_yz_orig = torch.tensor([
    [-1.74,  29.81,  20.45,  20.29],
    [-1.51, -24.73, -30.88, -31.66],
], dtype=torch.float64)

LAMBDA = 1.0
MAX_CYL_DELTA = 0.005
ACTION_SCALE = 0.05

def make_J_pinv(J):
    JT = J.T.to(torch.float64)
    I2 = torch.eye(2, dtype=torch.float64)
    return (JT @ torch.linalg.inv(J.to(torch.float64) @ JT + LAMBDA**2 * I2)).float()

# Variant definitions
VARIANTS = {
    "A_original": {
        "name": "A_original",
        "J": J_yz_orig.clone(),
        "preprocess": lambda a: a,
    },
    "B_flip_z_act": {
        "name": "B_flip_z_act",
        "J": J_yz_orig.clone(),
        "preprocess": lambda a: torch.stack([a[:, 0], -a[:, 1]], dim=-1),
    },
    "C_flip_z_row": {
        "name": "C_flip_z_row",
        "J": torch.stack([J_yz_orig[0], -J_yz_orig[1]]),
        "preprocess": lambda a: a,
    },
    "D_flip_y_act": {
        "name": "D_flip_y_act",
        "J": J_yz_orig.clone(),
        "preprocess": lambda a: torch.stack([-a[:, 0], a[:, 1]], dim=-1),
    },
    "E_swap_yz": {
        "name": "E_swap_yz",
        "J": J_yz_orig.clone(),
        "preprocess": lambda a: torch.stack([a[:, 1], a[:, 0]], dim=-1),
    },
    "F_empirical": {
        "name": "F_empirical",
        "J": J_yz_orig.clone(),  # will be overwritten by empirical measurement
        "preprocess": lambda a: a,
        "empirical": True,
    },
}

def build_env(device):
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    env = Stage1YZTrackingEnvWrapper(
        cfg, horizon_steps=int(args_cli.horizon_steps),
        action_scale_m=ACTION_SCALE, device=device,
        kp_scale=float(args_cli.kp_scale), kd_scale=float(args_cli.kd_scale),
        force_rate_limit=0.01, velocity_filter_beta=0.8,
        trajectory_mode=args_cli.trajectory_mode,
        target_hold_steps=int(args_cli.target_hold_steps),
        post_reset_settle_steps=int(args_cli.post_reset_settle_steps),
        backend=args_cli.backend,
        bias_calibration_time_s=float(args_cli.bias_calibration_time_s),
        action_smoothing_beta=0.8, action_rate_limit=0.1,
        reward_mode="distance_only",
    )
    env._target_y_range = float(args_cli.target_y_range)
    env._target_z_range = float(args_cli.target_z_range)
    return env

def test_action_sign(env, J_pinv, preprocess_fn, device):
    """Test all 4 action signs with a single forward pass."""
    B = env.num_envs
    results = {}
    ee0 = env._tcp_yz().detach().clone()

    for label, action_vec in [("+Y", torch.tensor([1.0, 0.0])),
                               ("-Y", torch.tensor([-1.0, 0.0])),
                               ("+Z", torch.tensor([0.0, 1.0])),
                               ("-Z", torch.tensor([0.0, -1.0]))]:
        a = action_vec.unsqueeze(0).expand(B, -1).to(device)
        a_proc = preprocess_fn(a)
        dc = torch.clamp((J_pinv.to(device) @ a_proc.T).T, -MAX_CYL_DELTA, MAX_CYL_DELTA)
        cqh = env._uw.robot.data.joint_pos[:, env._cyl_ids].clone()
        lim = env._uw.robot.data.soft_joint_pos_limits[:, env._cyl_ids]
        cqt = torch.clamp(cqh + dc, lim[..., 0], lim[..., 1])
        for _ in range(env._H):
            env._uw.set_debug_manual_cylinder_targets(cqt, tool_position_target=env._tool_target)
            env._env.step(torch.zeros((B, 5), device=device))
            env._pin_root()
        ee1 = env._tcp_yz()
        delta = (ee1 - ee0).mean(dim=0)
        expected_y = (1.0 if "+Y" in label else -1.0) if "Y" in label else 0.0
        expected_z = (1.0 if "+Z" in label else -1.0) if "Z" in label else 0.0
        y_ok = (delta[0] * expected_y > 0) if expected_y != 0 else True
        z_ok = (delta[1] * expected_z > 0) if expected_z != 0 else True
        results[label] = {"delta": delta.detach().cpu().tolist(), "pass": bool(y_ok and z_ok)}
        # Reset back
        env._uw._reset_idx(list(range(B)))
        for _ in range(int(args_cli.post_reset_settle_steps)):
            env._env.step(torch.zeros((B, 5), device=device))
        env._base_root_pose = env._uw.robot.data.root_state_w[:, :7].clone()
        ee0 = env._tcp_yz().detach().clone()
    return results

def test_greedy_rollout(env, J_pinv, preprocess_fn, device, num_rollouts):
    """Run greedy rollout and return mean distance."""
    B = env.num_envs
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

    # Override the mapper temporarily by monkey-patching
    # Save originals
    orig_J_pinv = env._J_pinv.clone()
    env._J_pinv = J_pinv.clone().to(device)

    dists = []
    jumps = []
    episodes = 0
    per_ep_dist = []
    per_ep_jump = []
    saved_max_ep = env.max_episode_length
    env.max_episode_length = 200

    while episodes < num_rollouts:
        err = env._target_yz - env._tcp_yz()
        # Apply preprocess to greedy action
        raw_greedy = torch.clamp(err / ACTION_SCALE, -1.0, 1.0)
        action = preprocess_fn(raw_greedy)
        # Pass through the standard step (which applies smoothing/rate limit)
        # But we need to bypass the internal J_pinv — use the overridden one
        obs, reward, done, extras = env.step(action)
        per_ep_dist.append(extras["log"]["distance"].item())
        per_ep_jump.append(extras["log"]["ee_jump"].item())
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.ndim == 0:
                done_ids = done_ids.unsqueeze(0)
            for _ in done_ids:
                if episodes >= num_rollouts:
                    break
                if per_ep_dist:
                    dists.append(float(np.mean(per_ep_dist)))
                    jumps.append(float(np.mean(per_ep_jump)))
                episodes += 1
            if episodes < num_rollouts:
                # Re-settle
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
            per_ep_dist = []
            per_ep_jump = []

    env.max_episode_length = saved_max_ep
    env._J_pinv = orig_J_pinv  # restore
    if not dists:
        return 0, 0, 0
    return float(np.mean(dists)), float(np.mean(jumps)), float(np.max(jumps))

print(f"[MAPPER] building env num_envs={args_cli.num_envs}", flush=True)
env = build_env(args_cli.device)
B = env.num_envs
device = args_cli.device

# Determine which variants to test
if args_cli.variants == "all":
    variant_keys = list(VARIANTS.keys())
else:
    variant_keys = [v.strip() for v in args_cli.variants.split(",")]

results = []

# First get zero baseline
print(f"\n[MAPPER] Zero hold baseline", flush=True)
env.reset()
for _ in range(30):
    env.step(torch.zeros((B, 2), device=device))
env._ep_length.zero_()
zero_dists = []
for _ in range(int(args_cli.num_rollouts)):
    env.step(torch.zeros((B, 2), device=device))
    zero_dists.append(float(torch.norm(env._tcp_yz() - env._target_yz, dim=-1).mean()))
zero_mean = float(np.mean(zero_dists)) if zero_dists else 0
print(f"  zero mean_distance = {zero_mean:.4f} m", flush=True)

for vkey in variant_keys:
    vdef = VARIANTS[vkey]
    print(f"\n{'='*50}", flush=True)
    print(f"[MAPPER] Testing variant: {vdef['name']}", flush=True)
    print(f"{'='*50}", flush=True)

    J = vdef["J"]
    preproc = vdef["preprocess"]
    J_pinv = make_J_pinv(J)

    # If empirical, measure response
    if vdef.get("empirical"):
        print(f"  Measuring empirical 2x2 response...", flush=True)
        # Simple empirical measurement: apply small +Y, +Z actions and measure EE delta
        env.reset()
        for _ in range(30):
            env.step(torch.zeros((B, 2), device=device))
        env._ep_length.zero_()
        ee0 = env._tcp_yz().mean(dim=0).detach()
        # +Y test
        for _ in range(env._H):
            env._uw.set_debug_manual_cylinder_targets(
                torch.clamp(env._uw.robot.data.joint_pos[:, env._cyl_ids] + 0.003,
                           env._uw.robot.data.soft_joint_pos_limits[:, env._cyl_ids, 0],
                           env._uw.robot.data.soft_joint_pos_limits[:, env._cyl_ids, 1]),
                tool_position_target=env._tool_target)
            env._env.step(torch.zeros((B, 5), device=device))
            env._pin_root()
        ee1 = env._tcp_yz().mean(dim=0).detach()
        # This is just a rough check — proper calibration needs more work
        delta_yz = (ee1 - ee0).cpu()
        print(f"  Empirical delta for small step: ({delta_yz[0]:.4f}, {delta_yz[1]:.4f})", flush=True)
        # For now, skip empirical since it needs careful implementation
        print(f"  F_empirical: SKIPPED (needs proper calibration)", flush=True)
        continue

    # Test action signs
    print(f"  Action sign tests:", flush=True)
    sign_results = test_action_sign(env, J_pinv, preproc, device)
    n_pass = sum(1 for r in sign_results.values() if r["pass"])
    for label, r in sign_results.items():
        status = "PASS" if r["pass"] else "FAIL"
        d = r["delta"]
        print(f"    {label}: delta=({d[0]:.5f},{d[1]:.5f}) {status}", flush=True)
    print(f"  Action sign: {n_pass}/4", flush=True)

    # Test greedy rollout
    print(f"  Greedy rollout ({args_cli.num_rollouts} rollouts)...", flush=True)
    gd, gj_mean, gj_max = test_greedy_rollout(env, J_pinv, preproc, device, int(args_cli.num_rollouts))
    impr = (zero_mean - gd) / max(zero_mean, 1e-8) * 100
    print(f"  greedy mean_distance = {gd:.4f} m", flush=True)
    print(f"  improvement over zero = {impr:.1f}%", flush=True)
    print(f"  mean_jump = {gj_mean:.4f} m  max_jump = {gj_max:.4f} m", flush=True)

    results.append({
        "variant": vdef["name"],
        "sign_pass": n_pass,
        "sign_details": {k: v["pass"] for k, v in sign_results.items()},
        "greedy_dist": gd,
        "improvement_pct": impr,
        "mean_jump": gj_mean,
        "max_jump": gj_max,
    })

# Print summary
print(f"\n{'='*60}", flush=True)
print(f"MAPPER VARIANT SUMMARY", flush=True)
print(f"zero mean_distance = {zero_mean:.4f} m", flush=True)
print(f"{'='*60}", flush=True)
for r in results:
    sel = ""
    if r["sign_pass"] >= 3 and r["improvement_pct"] > 10 and r["max_jump"] < 0.08:
        sel = " *** BEST ***"
    print(f"\n  {r['variant']}:", flush=True)
    print(f"    sign: {r['sign_pass']}/4 ({r['sign_details']})", flush=True)
    print(f"    greedy_dist={r['greedy_dist']:.4f}m  improvement={r['improvement_pct']:.1f}%", flush=True)
    print(f"    mean_jump={r['mean_jump']:.4f}m  max_jump={r['max_jump']:.4f}m{sel}", flush=True)

env.close()
simulation_app.close()
print("\n[MAPPER] done", flush=True)
