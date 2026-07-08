#!/usr/bin/env python3
"""
External evaluation script for Stage-1 adapter PPO.

Compares:
    - zero (no control)
    - random_ppo_action
    - adapter_greedy (proportional controller + adapter)
    - ppo_best (trained PPO + adapter)
    - ppo_final (final PPO + adapter)
"""

from __future__ import annotations
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 adapter PPO")
    parser.add_argument("--checkpoint", type=str, required=True, help="PPO checkpoint directory")
    parser.add_argument("--adapter_checkpoint", type=str, required=True, help="Adapter checkpoint")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--trajectory_mode", type=str, default="sequence_easy", help="Trajectory mode")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias", help="Lower controller")
    parser.add_argument("--kp_scale", type=float, default=0.1, help="PID Kp")
    parser.add_argument("--kd_scale", type=float, default=0.5, help="PID Kd")
    parser.add_argument("--ppo_action_scale", type=float, default=0.02, help="PPO action scale")
    return parser.parse_args()


def create_env(args):
    """Create environment."""
    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
    from kevin_integration.new_observations_pipeline.ppo1_adapter.stage1_adapter_env_wrapper import Stage1AdapterEnvWrapper

    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = args.num_envs

    if args.headless:
        cfg.viewer.eye = [0, 0, 0]

    env = Stage1AdapterEnvWrapper(
        cfg,
        adapter_checkpoint_path=args.adapter_checkpoint,
        horizon_steps=args.max_steps,
        ppo_action_scale=args.ppo_action_scale,
        device=args.device,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        backend=args.backend,
        trajectory_mode=args.trajectory_mode,
        target_hold_steps=50,
        post_reset_settle_steps=30,
    )

    return env


def evaluate_controller(env, controller_fn, num_episodes, max_steps):
    """Evaluate a controller."""
    episode_metrics = []

    for ep in tqdm(range(num_episodes), desc="Episodes"):
        obs, extras = env.reset()

        ep_distances = []
        ep_successes = []
        ep_jumps = []

        for step in range(max_steps):
            action = controller_fn(obs, extras)
            obs, rewards, dones, extras = env.step(action)

            distance = extras["distance"]
            success = distance < 0.10
            jump = extras["ee_jump"]

            ep_distances.append(distance.cpu())
            ep_successes.append(success.cpu())
            ep_jumps.append(jump.cpu())

        episode_metrics.append({
            "mean_distance": torch.cat(ep_distances).mean().item(),
            "success_rate": torch.cat(ep_successes).float().mean().item(),
            "mean_jump": torch.cat(ep_jumps).mean().item(),
            "max_jump": torch.cat(ep_jumps).max().item(),
        })

    # Aggregate
    return {
        "mean_distance": np.mean([m["mean_distance"] for m in episode_metrics]),
        "median_distance": np.median([m["mean_distance"] for m in episode_metrics]),
        "success_rate": np.mean([m["success_rate"] for m in episode_metrics]),
        "mean_jump": np.mean([m["mean_jump"] for m in episode_metrics]),
        "max_jump": np.max([m["max_jump"] for m in episode_metrics]),
    }


def main():
    args = parse_args()

    print("="*80)
    print("[evaluate_stage1_adapter_ppo] External Evaluation")
    print("="*80)

    # Create environment
    print("\n[1/4] Creating environment...")
    env = create_env(args)

    # Load PPO policy
    print("\n[2/4] Loading PPO policy...")
    from rsl_rl.modules import ActorCritic

    checkpoint_dir = Path(args.checkpoint)
    best_checkpoint = checkpoint_dir / "model_best.pt"
    final_checkpoint = checkpoint_dir / "model_final.pt"

    actor_critic = ActorCritic(
        num_obs=env.num_obs,
        num_privileged_obs=None,
        num_actions=env.num_actions,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation='elu',
        init_noise_std=1.0,
    ).to(args.device)

    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=args.device)
        actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        print(f"  Loaded best checkpoint: iter {checkpoint['iteration']}, dist {checkpoint['mean_distance']*1000:.1f}mm")
    else:
        print("  Warning: No best checkpoint found")

    actor_critic.eval()

    # Define controllers
    print("\n[3/4] Evaluating controllers...")

    controllers = {
        "zero": lambda obs, extras: torch.zeros((env.num_envs, env.num_actions), device=args.device),
        "random_ppo_action": lambda obs, extras: torch.randn((env.num_envs, env.num_actions), device=args.device) * 0.5,
        "adapter_greedy": lambda obs, extras: torch.clamp(0.5 * extras["target_error"] / args.ppo_action_scale, -1, 1),
        "ppo_best": lambda obs, extras: actor_critic.act_inference(obs),
    }

    results = {}
    for name, controller_fn in controllers.items():
        print(f"\n  Evaluating {name}...")
        results[name] = evaluate_controller(env, controller_fn, args.num_episodes, args.max_steps)

    # Print results
    print("\n[4/4] Results")
    print("="*80)

    for name in ["zero", "random_ppo_action", "adapter_greedy", "ppo_best"]:
        r = results[name]
        print(f"\n{name}:")
        print(f"  Mean distance: {r['mean_distance']*1000:.1f} mm")
        print(f"  Median distance: {r['median_distance']*1000:.1f} mm")
        print(f"  Success@0.10: {r['success_rate']*100:.1f}%")
        print(f"  Mean jump: {r['mean_jump']*1000:.1f} mm")
        print(f"  Max jump: {r['max_jump']*1000:.1f} mm")

    # Compare
    print("\n" + "="*80)
    print("Comparison:")
    greedy_dist = results["adapter_greedy"]["mean_distance"]
    ppo_dist = results["ppo_best"]["mean_distance"]

    if ppo_dist < greedy_dist * 0.95:
        print(f"✓ PPO beats greedy by {(1 - ppo_dist/greedy_dist)*100:.1f}%")
    elif ppo_dist < greedy_dist * 1.05:
        print(f"≈ PPO matches greedy (within 5%)")
    else:
        print(f"✗ PPO worse than greedy by {(ppo_dist/greedy_dist - 1)*100:.1f}%")

    print("="*80)

    env.close()


if __name__ == "__main__":
    main()
