#!/usr/bin/env python3
"""Deterministically compare the residual policy with its zero-action floor on the mapped arc."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "checkpoint",
    type=str,
    nargs="?",
    default="ID_FD_pipline/ppo1/checkpoints/best_ee_yz_residual.pt",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=17)
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
    cfg.scene.num_envs = int(args.num_envs)
    cfg.seed = int(args.seed)
    cfg.episode_length_s = 300.0
    cfg.enable_scripted_vehicle_motion = True
    cfg.device = args.device
    cfg.sim.device = args.device
    agent_cfg = IDFDEEResidualPPORunnerCfg()
    agent_cfg.device = args.device

    base_env = gym.make(RESIDUAL_PLAY_TASK_ID, cfg=cfg, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(Path(args.checkpoint).expanduser().resolve()))
        print(f"[RESIDUAL][EVAL] loaded {args.checkpoint}", flush=True)
        policy = runner.get_inference_policy(device=base_env.unwrapped.device)

        def rollout(use_policy: bool) -> float:
            base_env.reset(seed=int(args.seed))
            obs = env.get_observations()
            errors: list[float] = []
            total = int(base_env.unwrapped.cfg.residual_settle_steps) + int(args.steps)
            for step in range(total):
                with torch.inference_mode():
                    action = policy(obs) if use_policy else torch.zeros(
                        (env.num_envs, 3), device=base_env.unwrapped.device
                    )
                # Isaac's simulation buffers must not be advanced under
                # torch.inference_mode(); only the policy forward belongs there.
                obs_dict, _, terminated, truncated, _ = base_env.unwrapped.step(action)
                dones = (terminated | truncated).to(dtype=torch.long)
                obs = TensorDict(obs_dict, batch_size=[env.num_envs])
                policy.reset(dones)
                if step >= int(base_env.unwrapped.cfg.residual_settle_steps) + int(args.warmup_steps):
                    ee = base_env.unwrapped._compute_ee_pos_local()
                    error = torch.linalg.norm(
                        base_env.unwrapped._ee_goal_local - ee, dim=-1
                    )
                    errors.extend(error.detach().cpu().tolist())
            return float(np.mean(np.asarray(errors, dtype=np.float64)))

        passed = True
        zero_mean = rollout(use_policy=False)
        policy_mean = rollout(use_policy=True)
        passed = policy_mean < zero_mean
        print(
            f"[RESIDUAL][EVAL] mapped-arc: policy={policy_mean:.5f} "
            f"zero={zero_mean:.5f} beats_zero={passed}",
            flush=True,
        )
        print(f"[RESIDUAL][GATE] {'PASS' if passed else 'FAIL'}", flush=True)
        if not passed:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
