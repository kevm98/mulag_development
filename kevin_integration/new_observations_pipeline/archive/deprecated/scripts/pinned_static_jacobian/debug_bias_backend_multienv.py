#!/usr/bin/env python3
"""Multi-env bias backend smoke test: zero hold, +Z authority, action sign, greedy vs zero.

Run with:
  ./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/debug_bias_backend_multienv.py \
    --num_envs 16 --headless --device cuda:0 --backend explicit_effort_pid_bias \
    --kp_scale 0.1 --kd_scale 0.5 --max_cylinder_delta_m 0.005 \
    --post_reset_settle_steps 5 --bias_calibration_time_s 2.0 --debug_bias_shapes
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-env bias backend smoke test")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.005)
parser.add_argument("--post_reset_settle_steps", type=int, default=5)
parser.add_argument("--bias_calibration_time_s", type=float, default=2.0)
parser.add_argument("--debug_bias_shapes", action="store_true", default=False)
parser.add_argument("--target_y_range", type=float, default=0.03)
parser.add_argument("--target_z_range", type=float, default=0.01)
parser.add_argument("--target_hold_steps", type=int, default=999999)
parser.add_argument("--action_scale", type=float, default=0.05)
parser.add_argument("--horizon_steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage1_yz_tracking_env_wrapper import Stage1YZTrackingEnvWrapper

B = int(args_cli.num_envs)
device = args_cli.device

print(f"[SMOKE] num_envs={B} backend={args_cli.backend} kp={args_cli.kp_scale} kd={args_cli.kd_scale}", flush=True)


def build_env():
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = B
    env = Stage1YZTrackingEnvWrapper(
        cfg,
        horizon_steps=int(args_cli.horizon_steps),
        action_scale_m=float(args_cli.action_scale),
        device=device,
        kp_scale=float(args_cli.kp_scale),
        kd_scale=float(args_cli.kd_scale),
        force_rate_limit=0.01,
        velocity_filter_beta=0.8,
        trajectory_mode="sequence",
        target_hold_steps=int(args_cli.target_hold_steps),
        post_reset_settle_steps=int(args_cli.post_reset_settle_steps),
        backend=str(args_cli.backend),
        bias_calibration_time_s=float(args_cli.bias_calibration_time_s),
        debug_bias_shapes=bool(args_cli.debug_bias_shapes),
    )
    env._target_y_range = float(args_cli.target_y_range)
    env._target_z_range = float(args_cli.target_z_range)
    return env


def run_calibration_warmup(env):
    """Run bias calibration: let the env settle and calibrate over many steps."""
    env.reset()
    policy_dt = env._dt
    calib_steps = max(int(args_cli.bias_calibration_time_s / policy_dt) + 20, 30)
    print(f"[SMOKE] Running calibration warmup: {calib_steps} steps...", flush=True)
    for _ in range(calib_steps):
        env.step(torch.zeros((B, 2), device=device))
    print(f"[SMOKE] Calibration warmup complete.", flush=True)
    # Check calibration status
    uw = env._uw
    calib_count = uw._explicit_bias_calibrated.sum().item()
    calib_steps_arr = uw._explicit_bias_calib_step
    bias_vals = uw._explicit_force_bias
    print(f"[SMOKE] Bias calibrated: {calib_count}/{B} envs", flush=True)
    print(f"[SMOKE] Calib step range: [{calib_steps_arr.min().item()}, {calib_steps_arr.max().item()}]", flush=True)
    print(f"[SMOKE] Bias values (env0): {bias_vals[0].tolist()}", flush=True)
    print(f"[SMOKE] Bias norm (env0): {bias_vals[0].norm().item():.1f}", flush=True)


def zero_hold_test(env, steps=100):
    """Zero hold: measure jump with zero action. Does NOT reset — uses existing calibration."""
    # Settle without resetting
    for _ in range(args_cli.post_reset_settle_steps):
        env.step(torch.zeros((B, 2), device=device))
    ee = env._tcp_yz()
    env._target_yz = ee.clone()
    env._target_vel_yz.zero_()
    env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)

    all_jumps = []
    for _ in range(steps):
        _, _, _, extras = env.step(torch.zeros((B, 2), device=device))
        all_jumps.append(extras["log"]["ee_jump"].item())
    mean_jump = float(np.mean(all_jumps))
    max_jump = float(np.max(all_jumps))
    return mean_jump, max_jump


def z_authority_test(env):
    """+Z authority: compare progress for +Z, -Z, and zero actions. Does NOT reset."""
    # Settle without resetting
    for _ in range(args_cli.post_reset_settle_steps):
        env.step(torch.zeros((B, 2), device=device))

    z_range = float(args_cli.target_z_range)
    ee = env._tcp_yz()
    z_target = ee.clone()
    z_target[:, 1] += z_range

    results = {}
    for label, act in [("plus_z", torch.tensor([[0.0, 1.0]], device=device).expand(B, -1)),
                        ("minus_z", torch.tensor([[0.0, -1.0]], device=device).expand(B, -1)),
                        ("zero", torch.zeros((B, 2), device=device))]:
        # Settle without resetting
        for _ in range(args_cli.post_reset_settle_steps):
            env.step(torch.zeros((B, 2), device=device))
        env._target_yz = z_target.clone()
        env._target_vel_yz.zero_()
        env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)
        progresses = []
        distances = []
        for _ in range(10):
            _, _, _, extras = env.step(act)
            progresses.append(extras["log"]["progress"].item())
            distances.append(extras["log"]["distance"].item())
        results[label] = {
            "progress": float(np.mean(progresses)),
            "distance": float(np.mean(distances)),
        }
    return results


def action_sign_test(env):
    """Test all 4 action sign directions. Does NOT reset."""
    # Settle without resetting
    for _ in range(args_cli.post_reset_settle_steps):
        env.step(torch.zeros((B, 2), device=device))

    actions = [
        ("+Y", torch.tensor([1.0, 0.0], device=device)),
        ("-Y", torch.tensor([-1.0, 0.0], device=device)),
        ("+Z", torch.tensor([0.0, 1.0], device=device)),
        ("-Z", torch.tensor([0.0, -1.0], device=device)),
    ]
    passed = 0
    details = []
    for label, act in actions:
        # Settle without resetting
        for _ in range(args_cli.post_reset_settle_steps):
            env.step(torch.zeros((B, 2), device=device))
        ee_before = env._tcp_yz().detach().clone()
        for _ in range(3):
            env.step(act.unsqueeze(0).expand(B, -1))
        ee_after = env._tcp_yz().detach().clone()
        delta = (ee_after - ee_before).mean(dim=0)
        correct = False
        if label == "+Y" and delta[0] > 0:
            correct = True
        elif label == "-Y" and delta[0] < 0:
            correct = True
        elif label == "+Z" and delta[1] > 0:
            correct = True
        elif label == "-Z" and delta[1] < 0:
            correct = True
        if correct:
            passed += 1
        details.append(f"  {label}: delta=({delta[0].item():.5f},{delta[1].item():.5f}) {'PASS' if correct else 'FAIL'}")
    return passed, details


def greedy_vs_zero_rollouts(env, num_rollouts=10):
    """Compare greedy vs zero action over multiple rollouts."""
    greedy_dists = []
    zero_dists = []
    greedy_progs = []
    zero_progs = []
    greedy_jumps = []
    zero_jumps = []

    for _ in range(num_rollouts):
        # --- Greedy ---
        # Settle without resetting
        for _ in range(args_cli.post_reset_settle_steps):
            env.step(torch.zeros((B, 2), device=device))
        ee = env._tcp_yz()
        yr = float(args_cli.target_y_range)
        zr = float(args_cli.target_z_range)
        dy = (torch.rand(B, device=device) * 2 * yr - yr)
        dz = (torch.rand(B, device=device) * 2 * zr - zr)
        target = ee + torch.stack([dy, dz], dim=-1)
        env._target_yz = target.clone()
        env._target_vel_yz.zero_()
        env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)

        err_dir = target - ee
        err_dir = err_dir / (err_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        greedy_action = err_dir

        r_dists = []; r_progs = []; r_jumps = []
        for _ in range(20):
            _, _, _, extras = env.step(greedy_action)
            r_dists.append(extras["log"]["distance"].item())
            r_progs.append(extras["log"]["progress"].item())
            r_jumps.append(extras["log"]["ee_jump"].item())
        greedy_dists.append(float(np.mean(r_dists)))
        greedy_progs.append(float(np.mean(r_progs)))
        greedy_jumps.append(float(np.max(r_jumps)))

        # --- Zero ---
        # Settle without resetting
        for _ in range(args_cli.post_reset_settle_steps):
            env.step(torch.zeros((B, 2), device=device))
        env._target_yz = target.clone()
        env._target_vel_yz.zero_()
        env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)

        r_dists = []; r_progs = []; r_jumps = []
        for _ in range(20):
            _, _, _, extras = env.step(torch.zeros((B, 2), device=device))
            r_dists.append(extras["log"]["distance"].item())
            r_progs.append(extras["log"]["progress"].item())
            r_jumps.append(extras["log"]["ee_jump"].item())
        zero_dists.append(float(np.mean(r_dists)))
        zero_progs.append(float(np.mean(r_progs)))
        zero_jumps.append(float(np.max(r_jumps)))

    return {
        "greedy": {
            "mean_distance": float(np.mean(greedy_dists)),
            "mean_progress": float(np.mean(greedy_progs)),
            "max_jump": float(np.max(greedy_jumps)),
        },
        "zero": {
            "mean_distance": float(np.mean(zero_dists)),
            "mean_progress": float(np.mean(zero_progs)),
            "max_jump": float(np.max(zero_jumps)),
        },
    }


# =========================================================================
# MAIN
# =========================================================================
print(f"\n{'='*60}", flush=True)
print(f"BIAS BACKEND MULTI-ENV SMOKE TEST (num_envs={B})", flush=True)
print(f"{'='*60}", flush=True)

env = build_env()

# Test 1: Reset + calibration
print(f"\n--- Test 1: Reset + calibration ---", flush=True)
run_calibration_warmup(env)
print(f"  PASS: calibration completed without crash", flush=True)

for _ in range(20):
    env.step(torch.zeros((B, 2), device=device))

# Test 2: Bias shape report
print(f"\n--- Test 2: Bias shape report ---", flush=True)
uw = env._uw
print(f"  num_envs = {B}", flush=True)
print(f"  _explicit_force_bias.shape = {uw._explicit_force_bias.shape}", flush=True)
print(f"  _explicit_bias_calibrated.shape = {uw._explicit_bias_calibrated.shape}", flush=True)
print(f"  _explicit_bias_calib_step.shape = {uw._explicit_bias_calib_step.shape}", flush=True)
print(f"  _cylinder_pos_target.shape = {uw._cylinder_pos_target.shape}", flush=True)
print(f"  calib_step max = {uw._explicit_bias_calib_step.max().item()}", flush=True)
print(f"  calib_step min = {uw._explicit_bias_calib_step.min().item()}", flush=True)

# Test 3: Zero hold
print(f"\n--- Test 3: Zero hold (100 steps) ---", flush=True)
mean_jump, max_jump = zero_hold_test(env, steps=100)
zh_pass = mean_jump < 0.03 and max_jump < 0.06
print(f"  mean_jump = {mean_jump:.4f} m", flush=True)
print(f"  max_jump  = {max_jump:.4f} m", flush=True)
print(f"  Zero hold: {'PASS' if zh_pass else 'FAIL'}", flush=True)

# Test 4: +Z authority
print(f"\n--- Test 4: +Z authority ---", flush=True)
z_results = z_authority_test(env)
print(f"  +Z progress: {z_results['plus_z']['progress']:.4f}", flush=True)
print(f"  -Z progress: {z_results['minus_z']['progress']:.4f}", flush=True)
print(f"  zero progress: {z_results['zero']['progress']:.4f}", flush=True)
za_pass = (z_results["plus_z"]["progress"] > z_results["zero"]["progress"] and
           z_results["plus_z"]["progress"] > z_results["minus_z"]["progress"])
print(f"  +Z authority: {'PASS' if za_pass else 'FAIL'}", flush=True)

# Test 5: Action sign
print(f"\n--- Test 5: Action sign ---", flush=True)
sign_passed, sign_details = action_sign_test(env)
for d in sign_details:
    print(d, flush=True)
print(f"  Action sign: {sign_passed}/4 {'PASS' if sign_passed >= 3 else 'FAIL'}", flush=True)

# Test 6: Greedy vs zero
print(f"\n--- Test 6: Greedy vs zero (10 rollouts) ---", flush=True)
gvz = greedy_vs_zero_rollouts(env, num_rollouts=10)
print(f"  greedy mean_distance: {gvz['greedy']['mean_distance']:.4f} m", flush=True)
print(f"  zero   mean_distance: {gvz['zero']['mean_distance']:.4f} m", flush=True)
print(f"  greedy mean_progress: {gvz['greedy']['mean_progress']:.4f}", flush=True)
print(f"  zero   mean_progress: {gvz['zero']['mean_progress']:.4f}", flush=True)
print(f"  greedy max_jump: {gvz['greedy']['max_jump']:.4f} m", flush=True)
print(f"  zero   max_jump: {gvz['zero']['max_jump']:.4f} m", flush=True)
gvz_pass = (gvz["greedy"]["mean_distance"] < gvz["zero"]["mean_distance"] and
            gvz["greedy"]["mean_progress"] >= gvz["zero"]["mean_progress"] and
            gvz["greedy"]["max_jump"] < 0.06)
print(f"  Greedy vs zero: {'PASS' if gvz_pass else 'FAIL'}", flush=True)

# --- Summary ---
print(f"\n{'='*60}", flush=True)
print(f"SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Zero hold:      {'PASS' if zh_pass else 'FAIL'}", flush=True)
print(f"  +Z authority:   {'PASS' if za_pass else 'FAIL'}", flush=True)
print(f"  Action sign:    {sign_passed}/4", flush=True)
print(f"  Greedy vs zero: {'PASS' if gvz_pass else 'FAIL'}", flush=True)
overall = zh_pass and gvz_pass
print(f"  OVERALL:         {'PASS' if overall else 'FAIL'}", flush=True)

env.close()
simulation_app.close()
print("Done.", flush=True)
