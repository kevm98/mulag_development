#!/usr/bin/env python3
"""
Greedy Adapter Evaluation: Multi-step tracking with proportional controller.

Validation Gate 3: Greedy Tracking

Uses simple greedy upper command:
    desired_delta_ee = clamp(K * (target - current_ee), -max, +max)

Then:
    desired_delta_ee → adapter → cylinder_target → PhysX

Pass criteria:
    - Adapter beats zero baseline by > 25%
    - Adapter beats old Jacobian controller
    - Mean distance < 80-100mm
    - Success@0.10 > 80%
    - Mean jump < 30mm
    - Max jump < 80mm
"""

from __future__ import annotations
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Greedy adapter evaluation")
    parser.add_argument("--adapter_checkpoint", type=str, required=True, help="Path to trained adapter")
    parser.add_argument("--forward_checkpoint", type=str, default=None, help="Optional forward model")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--greedy_gain", type=float, default=0.5, help="Proportional gain K")
    parser.add_argument("--max_delta_ee", type=float, default=0.02, help="Max desired EE motion")
    parser.add_argument("--trajectory_mode", type=str, default="sequence_easy", help="Trajectory mode")
    parser.add_argument("--target_hold_steps", type=int, default=50, help="Target hold steps")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias", help="Lower controller")
    parser.add_argument("--kp_scale", type=float, default=0.1, help="PID Kp")
    parser.add_argument("--kd_scale", type=float, default=0.5, help="PID Kd")
    parser.add_argument("--bias_calibration_time_s", type=float, default=5.0, help="Bias calibration time")
    parser.add_argument("--settle_steps", type=int, default=30, help="Post-reset settle steps")
    return parser.parse_args()


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
        horizon_steps=args.max_steps,
        ppo_action_scale=1.0,  # We'll handle scaling manually
        device=args.device,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        backend=args.backend,
        bias_calibration_time_s=args.bias_calibration_time_s,
        trajectory_mode=args.trajectory_mode,
        target_hold_steps=args.target_hold_steps,
        post_reset_settle_steps=args.settle_steps,
        use_forward_model_validation=False,
    )

    return env


def evaluate_greedy_adapter(args):
    """Evaluate adapter with greedy proportional control."""
    print("="*80)
    print("[evaluate_adapter_greedy] Validation Gate 3: Greedy Tracking")
    print("="*80)

    # Create environment
    print("\n[1/3] Creating environment...")
    env = create_env(args, args.adapter_checkpoint, args.forward_checkpoint)

    # Storage
    episode_distances = []
    episode_successes = []
    episode_jumps = []
    episode_max_jumps = []
    episode_limit_violations = []

    # Run episodes
    print(f"\n[2/3] Running {args.num_episodes} episodes with greedy controller...")

    for ep in tqdm(range(args.num_episodes), desc="Episodes"):
        obs, extras = env.reset()

        ep_distances = []
        ep_jumps = []
        ep_successes = []
        ep_limit_viols = []

        for step in range(args.max_steps):
            # Greedy proportional control
            target_error = extras["target_error"]

            # Compute desired EE motion
            desired_delta_ee = args.greedy_gain * target_error
            desired_delta_ee = torch.clamp(
                desired_delta_ee,
                -args.max_delta_ee,
                args.max_delta_ee,
            )

            # Normalize for env (env expects action in [-1, 1])
            ppo_action = desired_delta_ee / env._ppo_action_scale

            # Step
            obs, rewards, dones, step_extras = env.step(ppo_action)
            extras = step_extras

            # Collect metrics
            distance = step_extras["distance"]
            jump = step_extras["ee_jump"]
            success = distance < 0.10

            ep_distances.append(distance.cpu())
            ep_jumps.append(jump.cpu())
            ep_successes.append(success.cpu())

        # Episode metrics
        episode_distances.append(torch.cat(ep_distances).mean().item())
        episode_successes.append(torch.cat(ep_successes).float().mean().item())
        episode_jumps.append(torch.cat(ep_jumps).mean().item())
        episode_max_jumps.append(torch.cat(ep_jumps).max().item())

    # Aggregate results
    mean_distance = np.mean(episode_distances)
    median_distance = np.median(episode_distances)
    success_rate_10cm = np.mean(episode_successes)
    mean_jump = np.mean(episode_jumps)
    max_jump = np.max(episode_max_jumps)

    print("\n[3/3] Results")
    print("="*80)
    print(f"  Mean distance: {mean_distance*1000:.1f} mm")
    print(f"  Median distance: {median_distance*1000:.1f} mm")
    print(f"  Success@0.10m: {success_rate_10cm*100:.1f}%")
    print(f"  Mean jump: {mean_jump*1000:.1f} mm")
    print(f"  Max jump: {max_jump*1000:.1f} mm")

    # Pass criteria
    distance_pass = mean_distance < 0.100  # 100mm
    success_pass = success_rate_10cm > 0.80
    safety_pass = mean_jump < 0.030 and max_jump < 0.080

    print("\nPass Criteria:")
    print(f"  Mean distance < 100mm: {'✓ PASS' if distance_pass else '✗ FAIL'}")
    print(f"  Success@0.10 > 80%: {'✓ PASS' if success_pass else '✗ FAIL'}")
    print(f"  Safety (mean<30mm, max<80mm): {'✓ PASS' if safety_pass else '✗ FAIL'}")

    print("\n" + "="*80)
    if distance_pass and success_pass and safety_pass:
        print("✓ PASS: Adapter validated. Proceed to PPO training.")
    else:
        print("✗ FAIL: Improve adapter before PPO training")
    print("="*80)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    evaluate_greedy_adapter(args)
