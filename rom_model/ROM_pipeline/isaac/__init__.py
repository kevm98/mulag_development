"""IsaacLab registrations for PPO -> shared controller -> ROM -> PhysX."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry


TASK_ID = "RRLAB-Mulag-ROM-CylinderSpace-v0"
PLAY_TASK_ID = "RRLAB-Mulag-ROM-CylinderSpace-PLAY-v0"


def _register_once(task_id: str, env_cfg: str) -> None:
    if task_id in registry:
        return
    gym.register(
        id=task_id,
        entry_point="PPO1_ID_FD_PhysX.tasks.cylinder_env:IDFDCylinderSpaceEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": "ROM_pipeline.isaac.env_cfg:ROMCylinderSpacePPORunnerCfg",
        },
    )


_register_once(TASK_ID, "ROM_pipeline.isaac.env_cfg:ROMCylinderSpaceEnvCfg")
_register_once(PLAY_TASK_ID, "ROM_pipeline.isaac.env_cfg:ROMCylinderSpaceEnvCfg_PLAY")

__all__ = ["PLAY_TASK_ID", "TASK_ID"]
