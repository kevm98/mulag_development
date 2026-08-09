"""Gym registrations for PPO2 -> command ROM -> PhysX tasks."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry


TASK_ID = "RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-v0"
PLAY_TASK_ID = "RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-PLAY-v0"
EE_TASK_ID = "RRLAB-Mulag-PPO2-ROM-PhysX-EETracking-v0"
EE_PLAY_TASK_ID = "RRLAB-Mulag-PPO2-ROM-PhysX-EETracking-PLAY-v0"
POLICY_V3_TASK_ID = "RRLAB-Mulag-PPO2-A-Policy-V3-v0"
POLICY_V3_PLAY_TASK_ID = "RRLAB-Mulag-PPO2-A-Policy-V3-PLAY-v0"


def _register_once(
    task_id: str,
    *,
    env_cfg: str,
    runner_cfg: str,
    entry_point: str = "mulag_plant.tasks.cylinder_env:PPO2ROMPhysXEnv",
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
    env_cfg="mulag_plant.tasks.env_cfg:PPO2ROMPhysXEnvCfg",
    runner_cfg="mulag_plant.tasks.env_cfg:PPO2ROMPhysXPPORunnerCfg",
)

_register_once(
    PLAY_TASK_ID,
    env_cfg="mulag_plant.tasks.env_cfg:PPO2ROMPhysXEnvCfg_PLAY",
    runner_cfg="mulag_plant.tasks.env_cfg:PPO2ROMPhysXPPORunnerCfg",
)

_register_once(
    EE_TASK_ID,
    env_cfg="mulag_plant.tasks.env_cfg:PPO2ROMPhysXEETrackingEnvCfg",
    runner_cfg="mulag_plant.tasks.env_cfg:PPO2ROMPhysXEETrackingPPORunnerCfg",
)

_register_once(
    EE_PLAY_TASK_ID,
    env_cfg="mulag_plant.tasks.env_cfg:PPO2ROMPhysXEETrackingEnvCfg_PLAY",
    runner_cfg="mulag_plant.tasks.env_cfg:PPO2ROMPhysXEETrackingPPORunnerCfg",
)

_register_once(
    POLICY_V3_TASK_ID,
    env_cfg="mulag_plant.policy_4cyl.env_cfg:FourCylinderResidualPolicyEnvCfg",
    runner_cfg=(
        "mulag_plant.policy_4cyl.env_cfg:"
        "FourCylinderResidualPolicyPPORunnerCfg"
    ),
    entry_point="mulag_plant.policy.env:ResidualPolicyEnv",
)

_register_once(
    POLICY_V3_PLAY_TASK_ID,
    env_cfg=(
        "mulag_plant.policy_4cyl.env_cfg:"
        "FourCylinderResidualPolicyEnvCfg_PLAY"
    ),
    runner_cfg=(
        "mulag_plant.policy_4cyl.env_cfg:"
        "FourCylinderResidualPolicyPPORunnerCfg"
    ),
    entry_point="mulag_plant.policy.env:ResidualPolicyEnv",
)


__all__ = [
    "TASK_ID", "PLAY_TASK_ID", "EE_TASK_ID", "EE_PLAY_TASK_ID",
    "POLICY_V3_TASK_ID", "POLICY_V3_PLAY_TASK_ID",
]
