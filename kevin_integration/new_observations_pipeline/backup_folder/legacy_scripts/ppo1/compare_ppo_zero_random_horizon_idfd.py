#!/usr/bin/env python3
"""Compare active PPO1 Horizon-IDFD control against zero/random baselines."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Compare PPO1 Horizon-IDFD against zero/random baselines.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=20)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--include_greedy", action="store_true")
parser.add_argument("--fast_exit", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""):
    args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import (
    HorizonIDFDEnv,
    HorizonIDFDRslRlVecEnvWrapper,
)
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import (
    MulagUpperLevelEnvCfg,
    MulagUpperLevelPPORunnerCfg,
)
from kevin_integration.new_observations_pipeline.ppo1.rewards_adapter import compute_tracking_error
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def build_env() -> HorizonIDFDEnv:
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.ee_action_mode = "task_space_delta"
    cfg.ee_action_dim = 3
    cfg.ppo_stage = "local_reach_stage0"
    cfg.stage0_level = 0
    cfg.lowlevel_actuator_backend = "explicit_effort_pid_bias"
    cfg.enable_scripted_vehicle_motion = False
    cfg.enable_startup_summary = False
    cfg.device = args_cli.device
    cfg.sim.device = args_cli.device
    apply_kevin_sim_memory_optimizations(cfg.sim)
    return HorizonIDFDEnv(
        base_cfg=cfg,
        horizon_steps=int(args_cli.horizon_steps),
        action_scale_m=0.01,
        fd_gains=[2.0, 3.0, 3.0, 3.0],
        device=args_cli.device,
    )


def finite_body_stats(env: HorizonIDFDEnv) -> tuple[int, float, int]:
    body_pos = env._uw.robot.data.body_link_state_w[:, :, :3]
    nan_inf_count = int((~torch.isfinite(body_pos)).sum().detach().cpu().item())
    max_abs_body = float(body_pos.abs().amax().detach().cpu().item())
    huge_count = int((body_pos.abs().amax(dim=(1, 2)) > 100.0).sum().detach().cpu().item())
    return nan_inf_count, max_abs_body, huge_count


def greedy_action(env: HorizonIDFDEnv) -> torch.Tensor:
    target = env._uw._command_pos_w[:, :3]
    ee = env._tcp_pos_base_frame()
    err = target - ee
    return torch.clamp(err / max(float(env.cfg.ee_action_scale_m), 1.0e-6), -1.0, 1.0)


def summarize(label: str, errors: list[float], rewards: list[float], invalid: int, max_body: float, huge: int) -> dict:
    first = errors[0] if errors else float("nan")
    last = errors[-1] if errors else float("nan")
    mean = sum(errors) / max(len(errors), 1)
    reward_mean = sum(rewards) / max(len(rewards), 1)
    improved = first - last
    print(
        f"{label}: first_error={first:.6f} last_error={last:.6f} "
        f"mean_error={mean:.6f} improvement={improved:.6f} "
        f"mean_reward={reward_mean:.6f} nan_inf={invalid} "
        f"max_abs_body_world_position={max_body:.6f} huge_transform_warnings={huge}",
        flush=True,
    )
    return {
        "label": label,
        "first": first,
        "last": last,
        "mean": mean,
        "improvement": improved,
        "invalid": invalid,
        "max_body": max_body,
        "huge": huge,
    }


def run_policy(env: HorizonIDFDEnv, label: str, action_fn) -> dict:
    env.reset()
    errors: list[float] = []
    rewards: list[float] = []
    invalid_total = 0
    max_body = 0.0
    huge_total = 0
    for _ in range(int(args_cli.num_steps)):
        action = action_fn(env)
        _, reward, terminated, truncated, _ = env.step(action)
        errors.append(float(compute_tracking_error(env._uw).mean().detach().cpu().item()))
        rewards.append(float(reward.mean().detach().cpu().item()))
        invalid, max_abs, huge = finite_body_stats(env)
        invalid_total += invalid
        max_body = max(max_body, max_abs)
        huge_total += huge
        if bool((terminated | truncated).any().detach().cpu().item()):
            env.reset()
    return summarize(label, errors, rewards, invalid_total, max_body, huge_total)


def run_checkpoint(checkpoint: str) -> dict:
    base_env = build_env()
    env = HorizonIDFDRslRlVecEnvWrapper(base_env)
    try:
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = MulagUpperLevelPPORunnerCfg()
        agent_cfg.device = args_cli.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=args_cli.device)
        env.reset()
        errors: list[float] = []
        rewards: list[float] = []
        invalid_total = 0
        max_body = 0.0
        huge_total = 0
        for _ in range(int(args_cli.num_steps)):
            with torch.no_grad():
                action = policy(env.get_observations())
            _, reward, done, _ = env.step(action)
            errors.append(float(compute_tracking_error(base_env._uw).mean().detach().cpu().item()))
            rewards.append(float(reward.mean().detach().cpu().item()))
            invalid, max_abs, huge = finite_body_stats(base_env)
            invalid_total += invalid
            max_body = max(max_body, max_abs)
            huge_total += huge
            if bool(done.any().detach().cpu().item()):
                env.reset()
        return summarize("ppo", errors, rewards, invalid_total, max_body, huge_total)
    finally:
        env.close()


try:
    print("PPO1 BASELINE COMPARISON", flush=True)
    base_env = build_env()
    zero = run_policy(base_env, "zero", lambda env: torch.zeros((env.num_envs, env.num_actions), device=env.device))
    random = run_policy(
        base_env,
        "random",
        lambda env: torch.empty((env.num_envs, env.num_actions), device=env.device).uniform_(-1.0, 1.0),
    )
    greedy = None
    if args_cli.include_greedy:
        greedy = run_policy(base_env, "greedy", greedy_action)
    ppo = None
    checkpoint = str(args_cli.checkpoint or "")
    if checkpoint:
        ppo = run_checkpoint(checkpoint)
    else:
        print("ppo: checkpoint not provided; learning verdict is NOT ASSESSED", flush=True)

    physics_ok = zero["invalid"] == random["invalid"] == 0 and zero["huge"] == random["huge"] == 0
    if greedy is not None:
        physics_ok = physics_ok and greedy["invalid"] == 0 and greedy["huge"] == 0
    if ppo is None:
        print(f"physics_valid_for_baselines: {physics_ok}", flush=True)
        print("learning_verdict: not assessed without a PPO checkpoint", flush=True)
    else:
        ppo_ok = ppo["invalid"] == 0 and ppo["huge"] == 0
        better_than_zero = ppo["last"] < zero["last"] and ppo["improvement"] > zero["improvement"]
        better_than_random = ppo["last"] < random["last"] and ppo["improvement"] > random["improvement"]
        learning = bool(physics_ok and ppo_ok and better_than_zero and better_than_random)
        print(f"physics_valid_for_comparison: {physics_ok and ppo_ok}", flush=True)
        print(f"ppo_improves_over_zero: {better_than_zero}", flush=True)
        print(f"ppo_improves_over_random: {better_than_random}", flush=True)
        print(f"learning_verdict: {'learning' if learning else 'not demonstrated'}", flush=True)
    if args_cli.fast_exit:
        os._exit(0)
finally:
    simulation_app.close()
