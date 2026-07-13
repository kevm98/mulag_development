#!/usr/bin/env python3
"""
Train Stage-1 PPO with frozen adapter.

PPO learns task-space control policy:
    - Observation: target error, target velocity, previous action (6D)
    - Action: Desired EE motion [ΔY, ΔZ] (2D)
    
The frozen adapter converts PPO action to cylinder commands.
"""

from __future__ import annotations
import argparse
import os
import torch
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage-1 PPO with frozen adapter")
    parser.add_argument("--adapter_checkpoint", type=str, required=True, help="Path to frozen adapter")
    parser.add_argument("--forward_checkpoint", type=str, default=None, help="Optional forward model")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=50, help="Max PPO iterations")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint interval")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--trajectory_mode", type=str, default="sequence_easy", help="Trajectory mode")
    parser.add_argument("--target_hold_steps", type=int, default=50, help="Target hold steps")
    parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias", help="Lower controller")
    parser.add_argument("--kp_scale", type=float, default=0.1, help="PID Kp")
    parser.add_argument("--kd_scale", type=float, default=0.5, help="PID Kd")
    parser.add_argument("--bias_calibration_time_s", type=float, default=5.0, help="Bias calibration time")
    parser.add_argument("--ppo_action_scale", type=float, default=0.02, help="PPO action scale (desired EE motion)")
    parser.add_argument("--adapter_max_delta_cylinder", type=float, default=0.0025, help="Adapter max output")
    parser.add_argument("--action_smoothing_beta", type=float, default=0.8, help="Action smoothing")
    parser.add_argument("--action_rate_limit", type=float, default=0.1, help="Action rate limit")
    parser.add_argument("--reward_mode", type=str, default="distance_only", help="Reward mode")
    parser.add_argument("--ppo_lr_scale", type=float, default=0.05, help="PPO learning rate scale")
    parser.add_argument("--ppo_clip_param", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--ppo_entropy_coef", type=float, default=0.001, help="PPO entropy coefficient")
    parser.add_argument("--ppo_num_learning_epochs", type=int, default=2, help="PPO learning epochs per iteration")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.experiment_name is None:
        args.experiment_name = f"ppo1_adapter_{args.trajectory_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("="*80)
    print(f"[train_stage1_adapter_ppo] Training PPO with frozen adapter")
    print(f"  Experiment: {args.experiment_name}")
    print("="*80)
    
    # Create environment
    print("\n[1/3] Creating environment...")
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
        adapter_checkpoint_path=args.adapter_checkpoint,
        forward_checkpoint_path=args.forward_checkpoint,
        horizon_steps=200,
        ppo_action_scale=args.ppo_action_scale,
        device=args.device,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        backend=args.backend,
        bias_calibration_time_s=args.bias_calibration_time_s,
        trajectory_mode=args.trajectory_mode,
        target_hold_steps=args.target_hold_steps,
        post_reset_settle_steps=30,
        reward_mode=args.reward_mode,
        action_smoothing_beta=args.action_smoothing_beta,
        action_rate_limit=args.action_rate_limit,
        adapter_max_delta_cylinder=args.adapter_max_delta_cylinder,
        use_forward_model_validation=False,
    )
    
    # Create PPO agent
    print("\n[2/3] Creating PPO agent...")
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms import PPO
    
    # PPO config
    from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
    
    ppo_cfg = PPO.Config()
    ppo_cfg.learning_rate = 1e-3 * args.ppo_lr_scale
    ppo_cfg.num_learning_epochs = args.ppo_num_learning_epochs
    ppo_cfg.num_mini_batches = 4
    ppo_cfg.clip_param = args.ppo_clip_param
    ppo_cfg.entropy_coef = args.ppo_entropy_coef
    ppo_cfg.value_loss_coef = 1.0
    ppo_cfg.gamma = 0.99
    ppo_cfg.lam = 0.95
    ppo_cfg.max_grad_norm = 1.0
    ppo_cfg.use_clipped_value_loss = True
    
    # Create runner
    log_dir = Path("logs/rsl_rl") / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Log directory: {log_dir}")
    
    # Simple runner setup
    from rsl_rl.env import VecEnv
    
    # Create actor-critic
    actor_critic = ActorCritic(
        num_obs=env.num_obs,
        num_privileged_obs=None,
        num_actions=env.num_actions,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation='elu',
        init_noise_std=1.0,
    ).to(args.device)
    
    ppo_algorithm = PPO(actor_critic, device=args.device, **ppo_cfg.__dict__)
    
    # Training loop
    print(f"\n[3/3] Training for {args.max_iterations} iterations...")
    
    obs, _ = env.reset()
    
    best_mean_distance = float("inf")
    
    for iteration in range(args.max_iterations):
        # Collect experience
        obs_buf = []
        actions_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []
        
        for step in range(200):  # Horizon
            with torch.no_grad():
                actions = ppo_algorithm.act(obs, None)
                values = ppo_algorithm.actor_critic.evaluate(obs, None)[1]
            
            next_obs, rewards, dones, extras = env.step(actions)
            
            obs_buf.append(obs)
            actions_buf.append(actions)
            rewards_buf.append(rewards)
            dones_buf.append(dones)
            values_buf.append(values)
            
            obs = next_obs
        
        # Compute returns and advantages
        obs_buf = torch.stack(obs_buf)
        actions_buf = torch.stack(actions_buf)
        rewards_buf = torch.stack(rewards_buf)
        dones_buf = torch.stack(dones_buf)
        values_buf = torch.stack(values_buf)
        
        # Update PPO
        ppo_algorithm.process_env_step(rewards_buf[-1], dones_buf[-1])
        ppo_algorithm.compute_returns(values_buf[-1])
        
        mean_policy_loss, mean_value_loss, mean_entropy = ppo_algorithm.update()
        
        # Log metrics
        mean_distance = extras["episode"]["mean_distance"].mean().item()
        success_rate = extras["episode"]["success_rate"].mean().item()
        ppo_action_norm = extras["episode"]["ppo_action_norm"].mean().item()
        adapter_output_norm = extras["episode"]["adapter_output_norm"].mean().item()
        
        print(f"Iter {iteration+1}/{args.max_iterations} | "
              f"Dist: {mean_distance*1000:.1f}mm | "
              f"Success: {success_rate*100:.1f}% | "
              f"PPO norm: {ppo_action_norm:.3f} | "
              f"Adapter norm: {adapter_output_norm:.3f} | "
              f"Policy loss: {mean_policy_loss:.4f}")
        
        # Save checkpoint
        if (iteration + 1) % args.save_interval == 0 or mean_distance < best_mean_distance:
            checkpoint_path = log_dir / f"model_{iteration+1}.pt"
            torch.save({
                "iteration": iteration + 1,
                "actor_critic_state_dict": actor_critic.state_dict(),
                "optimizer_state_dict": ppo_algorithm.optimizer.state_dict(),
                "mean_distance": mean_distance,
                "success_rate": success_rate,
            }, checkpoint_path)
            
            if mean_distance < best_mean_distance:
                best_mean_distance = mean_distance
                best_path = log_dir / "model_best.pt"
                torch.save({
                    "iteration": iteration + 1,
                    "actor_critic_state_dict": actor_critic.state_dict(),
                    "optimizer_state_dict": ppo_algorithm.optimizer.state_dict(),
                    "mean_distance": mean_distance,
                    "success_rate": success_rate,
                }, best_path)
                print(f"  → Saved best model (dist: {mean_distance*1000:.1f}mm)")
    
    print("\n[train_stage1_adapter_ppo] Training complete!")
    print(f"  Best mean distance: {best_mean_distance*1000:.1f} mm")
    
    env.close()


if __name__ == "__main__":
    main()
