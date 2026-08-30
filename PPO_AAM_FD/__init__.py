"""PPO pipeline: policy -> online causal AAM -> ID adapter -> frozen FD -> PhysX."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry

from PPO_ROM.gates import GATES, REQUIRED_MESHES, check_gates  # noqa: F401
from PPO_ROM.rewards import RewardWeights, compute_reward  # noqa: F401

TERRAIN_TASK_ID = "RRLAB-Mulag-Verge-ID-AAM-FD-v0"
TERRAIN_PLAY_TASK_ID = "RRLAB-Mulag-Verge-ID-AAM-FD-PLAY-v0"


def _register_once(task_id: str, *, entry_point: str, env_cfg: str, runner_cfg: str) -> None:
    if task_id in registry:
        return
    gym.register(
        id=task_id,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg, "rsl_rl_cfg_entry_point": runner_cfg},
    )


_register_once(
    TERRAIN_TASK_ID,
    entry_point="PPO_AAM_FD.env:PpoIDAAMFDEnv",
    env_cfg="PPO_AAM_FD.env_cfg:PpoIDAAMFDEnvCfg",
    runner_cfg="PPO_AAM_FD.env_cfg:PpoIDAAMFDPPORunnerCfg",
)
_register_once(
    TERRAIN_PLAY_TASK_ID,
    entry_point="PPO_AAM_FD.env:PpoIDAAMFDEnv",
    env_cfg="PPO_AAM_FD.env_cfg:PpoIDAAMFDEnvCfg_PLAY",
    runner_cfg="PPO_AAM_FD.env_cfg:PpoIDAAMFDPPORunnerCfg",
)

__all__ = ["TERRAIN_TASK_ID", "TERRAIN_PLAY_TASK_ID", "GATES", "REQUIRED_MESHES", "check_gates"]
