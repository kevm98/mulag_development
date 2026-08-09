"""PPO4_1: PPO3_1's verge task and reward over the ID/FD actuator.

Registers the gym task ids on import, mirroring PPO_ROM. The gate
contract and the reward are IMPORTED from PPO3_1 rather than copied -- a copy
would drift, and the moment it did, a PPO3_1 vs PPO4_1 comparison would be
quietly measuring that drift instead of the actuator.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry

# Re-exported so callers can use the same contract from either package without
# importing across pipelines by hand. These are the SAME objects, not copies.
from PPO_ROM.gates import GATES, REQUIRED_MESHES, check_gates  # noqa: F401
from PPO_ROM.rewards import RewardWeights, compute_reward  # noqa: F401


TERRAIN_TASK_ID = "RRLAB-Mulag-Verge-IDFD-v0"
TERRAIN_PLAY_TASK_ID = "RRLAB-Mulag-Verge-IDFD-PLAY-v0"


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
    entry_point="PPO_ID_FD.env:Ppo41IDFDEnv",
    env_cfg="PPO_ID_FD.env_cfg:Ppo41IDFDEnvCfg",
    runner_cfg="PPO_ID_FD.env_cfg:Ppo41IDFDPPORunnerCfg",
)
_register_once(
    TERRAIN_PLAY_TASK_ID,
    entry_point="PPO_ID_FD.env:Ppo41IDFDEnv",
    env_cfg="PPO_ID_FD.env_cfg:Ppo41IDFDEnvCfg_PLAY",
    runner_cfg="PPO_ID_FD.env_cfg:Ppo41IDFDPPORunnerCfg",
)


__all__ = [
    "TERRAIN_TASK_ID",
    "TERRAIN_PLAY_TASK_ID",
    "GATES",
    "REQUIRED_MESHES",
    "check_gates",
    "RewardWeights",
    "compute_reward",
]
