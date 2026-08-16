"""Gym registration for the final sole-controller PPO-ROM verge task."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry


TERRAIN_TASK_ID = "RRLAB-Mulag-Verge-ROM-v0"


def _register_once(
    task_id: str, *, entry_point: str, env_cfg: str, runner_cfg: str
) -> None:
    if task_id in registry:
        return
    gym.register(
        id=task_id,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": runner_cfg,
        },
    )


_register_once(
    TERRAIN_TASK_ID,
    entry_point="PPO_ROM.env:Ppo31SoleEnv",
    env_cfg="PPO_ROM.env_cfg:Ppo31SoleEnvCfg",
    runner_cfg="PPO_ROM.env_cfg:Ppo31PPORunnerCfg",
)
__all__ = ["TERRAIN_TASK_ID"]
