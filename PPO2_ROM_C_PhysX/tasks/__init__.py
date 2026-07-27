"""Gym registrations for PPO2-ROM_C."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry


TASK_ID = "RRLAB-Mulag-PPO2-ROM-C-AAM-Policy-v0"
PLAY_TASK_ID = "RRLAB-Mulag-PPO2-ROM-C-AAM-Policy-PLAY-v0"


def _register(task_id: str, env_cfg: str) -> None:
    if task_id in registry:
        return
    gym.register(
        id=task_id,
        entry_point="PPO2_ROM_C_PhysX.env:PPO2CAAMPolicyEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": (
                "PPO2_ROM_C_PhysX.env_cfg:PPO2CAAMPPORunnerCfg"
            ),
        },
    )


_register(TASK_ID, "PPO2_ROM_C_PhysX.env_cfg:PPO2CAAMPolicyEnvCfg")
_register(PLAY_TASK_ID, "PPO2_ROM_C_PhysX.env_cfg:PPO2CAAMPolicyEnvCfg_PLAY")

__all__ = ["TASK_ID", "PLAY_TASK_ID"]
