"""PPO4: sole-controller policy over the ID/FD actuator on the PPO3 verge.

Registers the gym task ids on import, mirroring PPO3_SOLE_PhysX.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry


TERRAIN_TASK_ID = "RRLAB-Mulag-PPO4-Verge-IDFD-v0"
TERRAIN_PLAY_TASK_ID = "RRLAB-Mulag-PPO4-Verge-IDFD-PLAY-v0"


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
    entry_point="PPO4_ID_FD_PhysX.terrain_env:TerrainIDFDEnv",
    env_cfg="PPO4_ID_FD_PhysX.terrain_env_cfg:TerrainIDFDEnvCfg",
    runner_cfg="PPO4_ID_FD_PhysX.terrain_env_cfg:TerrainIDFDPPORunnerCfg",
)
_register_once(
    TERRAIN_PLAY_TASK_ID,
    entry_point="PPO4_ID_FD_PhysX.terrain_env:TerrainIDFDEnv",
    env_cfg="PPO4_ID_FD_PhysX.terrain_env_cfg:TerrainIDFDEnvCfg_PLAY",
    runner_cfg="PPO4_ID_FD_PhysX.terrain_env_cfg:TerrainIDFDPPORunnerCfg",
)


__all__ = ["TERRAIN_TASK_ID", "TERRAIN_PLAY_TASK_ID"]
