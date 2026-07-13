#!/usr/bin/env python3
"""Train the stop-and-mow residual PPO policy.

Arm-only mowing with bounded residual corrections.  The deterministic state
machine handles phase transitions; PPO is active only during mowing.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Train residual PPO over the verified MULAG controller.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--preflight_only", action="store_true", default=False)
parser.add_argument("--debug_startup_stack", action="store_true", default=False)
parser.add_argument("--shape", type=str, default="arc", choices=["arc", "line", "bowed_curve", "sine_sweep"])
parser.add_argument("--speed", type=float, default=0.05, help="Path speed [m/s]")
parser.add_argument("--span", type=float, default=None, help="Lateral span [m]")
parser.add_argument("--min_path", type=float, default=5.0, help="Min accumulated cutter path [m]")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.debug_startup_stack:
    faulthandler.enable()
    faulthandler.dump_traceback_later(60.0, repeat=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from ID_FD_pipline.tasks import RESIDUAL_TASK_ID  # noqa: E402,F401
from ID_FD_pipline.ppo1.env_cfg import (  # noqa: E402
    IDFDEEResidualEnvCfg,
    IDFDEEResidualPPORunnerCfg,
)


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contains non-finite values")


def main() -> None:
    task_id = RESIDUAL_TASK_ID
    env_cfg = IDFDEEResidualEnvCfg()
    agent_cfg = IDFDEEResidualPPORunnerCfg()

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device
        env_cfg.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)
        agent_cfg.seed = int(args_cli.seed)

    total_iters = int(args_cli.max_iterations) if args_cli.max_iterations is not None else int(agent_cfg.max_iterations)
    agent_cfg.max_iterations = total_iters

    # Apply CLI overrides
    env_cfg.residual_mow_shape = args_cli.shape
    env_cfg.residual_path_speed_mps = float(args_cli.speed)
    env_cfg.residual_min_accumulated_path_m = float(args_cli.min_path)
    if args_cli.span is not None:
        env_cfg.residual_lateral_span_m = float(args_cli.span)

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, run_stamp))

    print(
        f"[ID_FD] train task={task_id} envs={env_cfg.scene.num_envs} "
        f"obs={env_cfg.observation_space} action={env_cfg.action_space} "
        f"shape={env_cfg.residual_mow_shape} speed={env_cfg.residual_path_speed_mps} m/s "
        f"span={env_cfg.residual_lateral_span_m}m min_path={env_cfg.residual_min_accumulated_path_m}m "
        f"iters={total_iters} log_dir={log_dir}",
        flush=True,
    )
    print(
        f"[ID_FD] models id={env_cfg.id_cylinder_checkpoint} fd={env_cfg.fd_cylinder_checkpoint}",
        flush=True,
    )

    print("[ID_FD] creating IsaacLab env", flush=True)
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    print("[ID_FD] env created", flush=True)
    try:
        if args_cli.preflight_only or total_iters <= 0:
            print("[ID_FD] reset check", flush=True)
            obs, _ = env.reset()
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            _finite("reset observation", policy_obs)
            expected_shape = (env.unwrapped.num_envs, int(env_cfg.observation_space))
            if tuple(policy_obs.shape) != expected_shape:
                raise RuntimeError(
                    f"policy observation shape must be {expected_shape}, "
                    f"got {tuple(policy_obs.shape)}"
                )
            print("[ID_FD] preflight passed", flush=True)
            return

        print("[ID_FD] wrapping env for RSL-RL", flush=True)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print("[ID_FD] creating PPO runner", flush=True)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        if args_cli.checkpoint:
            print(f"[ID_FD] loading checkpoint {args_cli.checkpoint}", flush=True)
            runner.load(args_cli.checkpoint)
        else:
            output_layer = next(
                module for module in reversed(runner.alg.actor.mlp)
                if isinstance(module, torch.nn.Linear)
            )
            torch.nn.init.zeros_(output_layer.weight)
            torch.nn.init.zeros_(output_layer.bias)
            print("[ID_FD] residual actor initialized to exact zero-action baseline", flush=True)

        print("[ID_FD] starting arm-only PPO learn", flush=True)
        runner.learn(
            num_learning_iterations=total_iters, init_at_random_ep_len=False
        )
    finally:
        env.close()


if __name__ == "__main__":
    main_finished = False
    try:
        main()
        main_finished = True
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if main_finished or not args_cli.debug_startup_stack:
            simulation_app.close()
        else:
            print("[ID_FD] debug mode: skipped SimulationApp.close() after startup failure", flush=True)
