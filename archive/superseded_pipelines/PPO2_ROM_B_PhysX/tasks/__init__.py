"""Gym registrations for PPO2-ROM_B."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry


TASK_ID = "RRLAB-Mulag-PPO2-ROM-B-Robust-Policy-v0"
PLAY_TASK_ID = "RRLAB-Mulag-PPO2-ROM-B-Robust-Policy-PLAY-v0"


def _register(task_id: str, env_cfg: str) -> None:
    if task_id in registry:
        return
    gym.register(
        id=task_id,
        entry_point="PPO2_ROM_B_PhysX.env:PPO2ROMBPolicyEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": (
                "PPO2_ROM_B_PhysX.env_cfg:PPO2ROMBPPORunnerCfg"
            ),
        },
    )


_register(TASK_ID, "PPO2_ROM_B_PhysX.env_cfg:PPO2ROMBPolicyEnvCfg")
_register(PLAY_TASK_ID, "PPO2_ROM_B_PhysX.env_cfg:PPO2ROMBPolicyEnvCfg_PLAY")

__all__ = ["TASK_ID", "PLAY_TASK_ID"]

