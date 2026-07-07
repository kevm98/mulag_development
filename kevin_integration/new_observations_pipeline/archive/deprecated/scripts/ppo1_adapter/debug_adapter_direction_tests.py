#!/usr/bin/env python3
"""
Direction Tests: One-step closed-loop validation in Isaac.

Validation Gate 2: Direction Tests

Tests adapter with unit commands in each direction:
    +Y, -Y, +Z, -Z, diagonals

Pass criteria:
    - 4/4 axis tests pass (correct sign)
    - 3/4 diagonal tests pass
    - Mean jump < 30mm
    - Max jump < 80mm
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import torch
import numpy as np

# Parse args before importing IsaacLab
parser = argparse.ArgumentParser(description="Direction tests for adapter")
parser.add_argument("--adapter_checkpoint", type=str, required=True, help="Path to trained adapter")
parser.add_argument("--forward_checkpoint", type=str, default=None, help="Optional forward model")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--num_tests_per_direction", type=int, default=20, help="Tests per direction")
parser.add_argument("--ppo_action_scale", type=float, default=0.02, help="Desired EE motion magnitude")
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias", help="Lower controller")
parser.add_argument("--kp_scale", type=float, default=0.1, help="PID Kp")
parser.add_argument("--kd_scale", type=float, default=0.5, help="PID Kd")
parser.add_argument("--bias_calibration_time_s", type=float, default=5.0, help="Bias calibration time")
parser.add_argument("--settle_steps", type=int, default=300, help="Post-reset settle steps")
parser.add_argument("--num_steps_per_test", type=int, default=10, help="Number of steps to apply command per test")

# Add IsaacLab launcher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args.headless = True  # Force headless
args.device = args.device if hasattr(args, 'device') else "cuda:0"

# Launch IsaacLab app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def create_env(args, adapter_path, forward_path=None):
    """Create adapter environment."""
    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
    from kevin_integration.new_observations_pipeline.ppo1_adapter.stage1_adapter_env_wrapper import (
        Stage1AdapterEnvWrapper,
    )

    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = args.num_envs

    if args.headless:
        cfg.viewer.eye = [0, 0, 0]

    env = Stage1AdapterEnvWrapper(
        cfg,
        adapter_checkpoint_path=adapter_path,
        forward_checkpoint_path=forward_path,
        horizon_steps=10,
        ppo_action_scale=args.ppo_action_scale,
        device=args.device,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        backend=args.backend,
        bias_calibration_time_s=args.bias_calibration_time_s,
        post_reset_settle_steps=args.settle_steps,
        use_forward_model_validation=False,
        debug_logging=True,  # Enable debug logging to trace command flow
    )

    return env


def test_direction(env, direction_name, desired_delta_ee, num_tests, num_steps=10):
    """Test adapter response to a specific direction command over multiple steps."""
    results = []

    for test_idx in range(num_tests):
        # Reset
        obs, extras = env.reset()

        # Record initial EE
        initial_ee = extras["current_ee_yz"].clone()

        # Apply command for multiple steps
        ppo_action = desired_delta_ee.repeat(env.num_envs, 1)

        for step in range(num_steps):
            obs, rewards, dones, step_extras = env.step(ppo_action)

        # Final measurement
        final_ee = step_extras["current_ee_yz"]
        measured_delta = final_ee - initial_ee

        # Debug: print first test
        if test_idx == 0:
            mean_delta = measured_delta.mean(dim=0)
            print(f"    [{direction_name}] {num_steps}-step motion: {mean_delta.cpu().numpy()*1000} mm")

        # Compute metrics (jump per step)
        jump_per_step = measured_delta.norm(dim=-1) / num_steps

        results.append({
            "desired": desired_delta_ee.cpu() * num_steps,  # Total desired over all steps
            "measured": measured_delta.cpu(),
            "jump": jump_per_step.cpu(),
        })

    # Aggregate results
    all_desired = torch.stack([r["desired"] for r in results])
    all_measured = torch.cat([r["measured"] for r in results])
    all_jumps = torch.cat([r["jump"] for r in results])

    # Sign matching
    desired_sign = torch.sign(all_desired[0])
    measured_sign = torch.sign(all_measured.mean(dim=0))
    sign_match_y = (desired_sign[0] == measured_sign[0]) if desired_sign[0] != 0 else True
    sign_match_z = (desired_sign[1] == measured_sign[1]) if desired_sign[1] != 0 else True

    return {
        "direction": direction_name,
        "desired": all_desired[0].numpy(),
        "measured_mean": all_measured.mean(dim=0).numpy(),
        "measured_std": all_measured.std(dim=0).numpy(),
        "jump_mean": all_jumps.mean().item(),
        "jump_max": all_jumps.max().item(),
        "sign_match_y": sign_match_y,
        "sign_match_z": sign_match_z,
    }


def main(args):
    print("="*80)
    print("[debug_adapter_direction_tests] Validation Gate 2: Direction Tests")
    print("="*80)

    # Create environment
    print("\n[1/3] Creating environment...")
    env = create_env(args, args.adapter_checkpoint, args.forward_checkpoint)

    # Define test directions
    scale = args.ppo_action_scale
    directions = {
        "+Y": torch.tensor([scale, 0.0], device=args.device),
        "-Y": torch.tensor([-scale, 0.0], device=args.device),
        "+Z": torch.tensor([0.0, scale], device=args.device),
        "-Z": torch.tensor([0.0, -scale], device=args.device),
        "+Y+Z": torch.tensor([scale, scale], device=args.device) / np.sqrt(2),
        "+Y-Z": torch.tensor([scale, -scale], device=args.device) / np.sqrt(2),
        "-Y+Z": torch.tensor([-scale, scale], device=args.device) / np.sqrt(2),
        "-Y-Z": torch.tensor([-scale, -scale], device=args.device) / np.sqrt(2),
    }

    # Run tests
    print(f"\n[2/3] Running direction tests ({args.num_tests_per_direction} tests × {args.num_steps_per_test} steps per direction)...")

    all_results = []
    for direction_name, desired_delta in directions.items():
        print(f"  Testing {direction_name}...")
        result = test_direction(env, direction_name, desired_delta, args.num_tests_per_direction, args.num_steps_per_test)
        all_results.append(result)

    # Print results
    print("\n[3/3] Results")
    print("="*80)

    axis_tests = all_results[:4]  # +Y, -Y, +Z, -Z
    diagonal_tests = all_results[4:]  # Diagonals

    print("\nAxis Tests:")
    axis_pass_count = 0
    for r in axis_tests:
        sign_y_str = "✓" if r["sign_match_y"] else "✗"
        sign_z_str = "✓" if r["sign_match_z"] else "✗"
        axis_pass = r["sign_match_y"] and r["sign_match_z"]
        axis_pass_count += axis_pass

        print(f"  {r['direction']:>4s}: Desired [{r['desired'][0]:+.3f}, {r['desired'][1]:+.3f}] "
              f"→ Measured [{r['measured_mean'][0]:+.3f}±{r['measured_std'][0]:.3f}, "
              f"{r['measured_mean'][1]:+.3f}±{r['measured_std'][1]:.3f}] "
              f"| Signs: Y{sign_y_str} Z{sign_z_str} | Jump: {r['jump_mean']*1000:.1f}mm")

    print(f"\n  Axis tests passed: {axis_pass_count}/4")

    print("\nDiagonal Tests:")
    diagonal_pass_count = 0
    for r in diagonal_tests:
        sign_y_str = "✓" if r["sign_match_y"] else "✗"
        sign_z_str = "✓" if r["sign_match_z"] else "✗"
        diagonal_pass = r["sign_match_y"] and r["sign_match_z"]
        diagonal_pass_count += diagonal_pass

        print(f"  {r['direction']:>6s}: Desired [{r['desired'][0]:+.3f}, {r['desired'][1]:+.3f}] "
              f"→ Measured [{r['measured_mean'][0]:+.3f}±{r['measured_std'][0]:.3f}, "
              f"{r['measured_mean'][1]:+.3f}±{r['measured_std'][1]:.3f}] "
              f"| Signs: Y{sign_y_str} Z{sign_z_str} | Jump: {r['jump_mean']*1000:.1f}mm")

    print(f"\n  Diagonal tests passed: {diagonal_pass_count}/4")

    # Safety metrics
    all_jumps_mean = np.mean([r['jump_mean'] for r in all_results])
    all_jumps_max = np.max([r['jump_max'] for r in all_results])

    print(f"\nSafety Metrics:")
    print(f"  Mean jump: {all_jumps_mean*1000:.1f} mm")
    print(f"  Max jump: {all_jumps_max*1000:.1f} mm")

    # Pass criteria
    axis_pass = axis_pass_count == 4
    diagonal_pass = diagonal_pass_count >= 3
    safety_pass = all_jumps_mean < 0.030 and all_jumps_max < 0.080

    print("\nPass Criteria:")
    print(f"  Axis tests (4/4): {'✓ PASS' if axis_pass else '✗ FAIL'}")
    print(f"  Diagonal tests (3/4): {'✓ PASS' if diagonal_pass else '✗ FAIL'}")
    print(f"  Safety (mean<30mm, max<80mm): {'✓ PASS' if safety_pass else '✗ FAIL'}")

    print("\n" + "="*80)
    if axis_pass and diagonal_pass and safety_pass:
        print("✓ PASS: Proceed to Gate 3 (Greedy Evaluation)")
    else:
        print("✗ FAIL: Improve adapter before greedy evaluation")
    print("="*80)

    env.close()


if __name__ == "__main__":
    main(args)
    simulation_app.close()
