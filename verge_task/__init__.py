"""PPO3: sole-controller policy over ROM-A.

Registers the gym task ids on import, mirroring mulag_plant.tasks.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry


TASK_ID = "RRLAB-Mulag-PPO3-Sole-ROM-A-v0"
PLAY_TASK_ID = "RRLAB-Mulag-PPO3-Sole-ROM-A-PLAY-v0"
TERRAIN_TASK_ID = "RRLAB-Mulag-PPO3-Verge-ROM-A-v0"
TERRAIN_PLAY_TASK_ID = "RRLAB-Mulag-PPO3-Verge-ROM-A-PLAY-v0"

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
    TASK_ID,
    entry_point="verge_task.env:SolePolicyEnv",
    env_cfg="verge_task.env_cfg:SolePolicyEnvCfg",
    runner_cfg="verge_task.env_cfg:SolePolicyPPORunnerCfg",
)
_register_once(
    PLAY_TASK_ID,
    entry_point="verge_task.env:SolePolicyEnv",
    env_cfg="verge_task.env_cfg:SolePolicyEnvCfg_PLAY",
    runner_cfg="verge_task.env_cfg:SolePolicyPPORunnerCfg",
)
_register_once(
    TERRAIN_TASK_ID,
    entry_point="verge_task.terrain_env:TerrainSolePolicyEnv",
    env_cfg="verge_task.terrain_env_cfg:TerrainSolePolicyEnvCfg",
    runner_cfg="verge_task.terrain_env_cfg:TerrainSolePolicyPPORunnerCfg",
)
_register_once(
    TERRAIN_PLAY_TASK_ID,
    entry_point="verge_task.terrain_env:TerrainSolePolicyEnv",
    env_cfg="verge_task.terrain_env_cfg:TerrainSolePolicyEnvCfg_PLAY",
    runner_cfg="verge_task.terrain_env_cfg:TerrainSolePolicyPPORunnerCfg",
)


__all__ = ["TASK_ID", "PLAY_TASK_ID", "TERRAIN_TASK_ID", "TERRAIN_PLAY_TASK_ID"]
