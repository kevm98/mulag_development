"""Gym registrations for the cylinder-space ID-FD task."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import registry


TASK_ID = "Kevin-Mulag-IDFD-CylinderSpace-v0"
PLAY_TASK_ID = "Kevin-Mulag-IDFD-CylinderSpace-PLAY-v0"
EE_TASK_ID = "Kevin-Mulag-IDFD-EETracking-v0"
EE_PLAY_TASK_ID = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
POLICY_V3_TASK_ID = "RRLAB-Mulag-PPO1-A-Policy-V3-v0"
POLICY_V3_PLAY_TASK_ID = "RRLAB-Mulag-PPO1-A-Policy-V3-PLAY-v0"


def _register_once(
    task_id: str,
    *,
    env_cfg: str,
    runner_cfg: str,
    entry_point: str = "PPO1_ID_FD_PhysX.tasks.cylinder_env:IDFDCylinderSpaceEnv",
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
    env_cfg="PPO1_ID_FD_PhysX.tasks.env_cfg:IDFDCylinderSpaceEnvCfg",
    runner_cfg="PPO1_ID_FD_PhysX.tasks.env_cfg:IDFDCylinderSpacePPORunnerCfg",
)

_register_once(
    PLAY_TASK_ID,
    env_cfg="PPO1_ID_FD_PhysX.tasks.env_cfg:IDFDCylinderSpaceEnvCfg_PLAY",
    runner_cfg="PPO1_ID_FD_PhysX.tasks.env_cfg:IDFDCylinderSpacePPORunnerCfg",
)

_register_once(
    EE_TASK_ID,
    env_cfg="PPO1_ID_FD_PhysX.tasks.env_cfg:IDFDEETrackingEnvCfg",
    runner_cfg="PPO1_ID_FD_PhysX.tasks.env_cfg:IDFDEETrackingPPORunnerCfg",
)

_register_once(
    EE_PLAY_TASK_ID,
    env_cfg="PPO1_ID_FD_PhysX.tasks.env_cfg:IDFDEETrackingEnvCfg_PLAY",
    runner_cfg="PPO1_ID_FD_PhysX.tasks.env_cfg:IDFDEETrackingPPORunnerCfg",
)

_register_once(
    POLICY_V3_TASK_ID,
    env_cfg=(
        "PPO1_ID_FD_PhysX.policy_4cyl.env_cfg:"
        "FourCylinderResidualPolicyEnvCfg"
    ),
    runner_cfg=(
        "PPO1_ID_FD_PhysX.policy_4cyl.env_cfg:"
        "FourCylinderResidualPolicyPPORunnerCfg"
    ),
    entry_point="PPO1_ID_FD_PhysX.policy.env:ResidualPolicyEnv",
)

_register_once(
    POLICY_V3_PLAY_TASK_ID,
    env_cfg=(
        "PPO1_ID_FD_PhysX.policy_4cyl.env_cfg:"
        "FourCylinderResidualPolicyEnvCfg_PLAY"
    ),
    runner_cfg=(
        "PPO1_ID_FD_PhysX.policy_4cyl.env_cfg:"
        "FourCylinderResidualPolicyPPORunnerCfg"
    ),
    entry_point="PPO1_ID_FD_PhysX.policy.env:ResidualPolicyEnv",
)


__all__ = [
    "TASK_ID", "PLAY_TASK_ID", "EE_TASK_ID", "EE_PLAY_TASK_ID",
    "POLICY_V3_TASK_ID", "POLICY_V3_PLAY_TASK_ID",
]
