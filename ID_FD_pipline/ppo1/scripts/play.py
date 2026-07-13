#!/usr/bin/env python3
"""Visualize the moving-truck EE controller with optional residual RL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkpoint",
    default="ID_FD_pipline/ppo1/checkpoints/best_ee_yz_residual.pt",
    help="Residual PPO checkpoint.",
)
parser.add_argument("--steps", type=int, default=3600)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--controller_only", action="store_true", help="Use zero residual for the classical baseline.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from ID_FD_pipline.tasks import RESIDUAL_PLAY_TASK_ID  # noqa: E402,F401
from ID_FD_pipline.ppo1.env_cfg import (  # noqa: E402
    IDFDEEResidualEnvCfg_PLAY,
    IDFDEEResidualPPORunnerCfg,
)


def main() -> None:
    cfg = IDFDEEResidualEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.seed = int(args.seed)
    cfg.episode_length_s = max(300.0, float(args.steps) * cfg.decimation * cfg.sim.dt + 10.0)
    cfg.enable_scripted_vehicle_motion = True
    cfg.device = args.device
    cfg.sim.device = args.device

    agent_cfg = IDFDEEResidualPPORunnerCfg()
    agent_cfg.device = args.device
    base_env = gym.make(RESIDUAL_PLAY_TASK_ID, cfg=cfg, render_mode="human")
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    try:
        policy = None
        if not args.controller_only:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            checkpoint = str(Path(args.checkpoint).expanduser().resolve())
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=base_env.unwrapped.device)
            print(f"[PLAY] residual policy={checkpoint}", flush=True)
        else:
            print("[PLAY] controller-only baseline (zero residual)", flush=True)
        print("[PLAY] mapped-arc nominal; truck speed is a policy action (zero = parked)", flush=True)

        base_env.reset(seed=int(args.seed))
        obs = env.get_observations()
        for step in range(int(args.steps)):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                action = (
                    policy(obs)
                    if policy is not None
                    else torch.zeros((1, 3), device=base_env.unwrapped.device)
                )
            obs_dict, _, terminated, truncated, _ = base_env.unwrapped.step(action)
            dones = (terminated | truncated).to(dtype=torch.long)
            obs = TensorDict(obs_dict, batch_size=[1])
            if policy is not None:
                policy.reset(dones)
            if step % 300 == 0:
                ee = base_env.unwrapped._compute_ee_pos_local()
                error = torch.linalg.norm(
                    base_env.unwrapped._ee_goal_local[:, 1:3] - ee[:, 1:3], dim=-1
                )
                print(f"[PLAY] step={step:04d} yz_error={float(error.item()):.4f} m", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
