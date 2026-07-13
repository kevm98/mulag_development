from __future__ import annotations

import gymnasium as gym

gym.register(
    id="Kevin-Mulag-HydraulicROM-TaskSpace-v0",
    entry_point="kevin_integration.tasks.mulag_adaptive_rl_env:MulagAdaptiveRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagHydraulicROMTaskSpaceEnvCfg",
        "rsl_rl_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagHydraulicROMTaskSpacePPORunnerCfg",
    },
)

gym.register(
    id="Kevin-Mulag-AdaptiveRL-JointReach-v0",
    entry_point="kevin_integration.tasks.mulag_adaptive_rl_env:MulagAdaptiveRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagHydraulicROMTaskSpaceEnvCfg",
        "rsl_rl_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagHydraulicROMTaskSpacePPORunnerCfg",
    },
)

gym.register(
    id="Kevin-Mulag-AdaptiveRL-v0",
    entry_point="kevin_integration.tasks.mulag_adaptive_rl_env:MulagAdaptiveRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagHydraulicROMTaskSpaceEnvCfg",
        "rsl_rl_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagHydraulicROMTaskSpacePPORunnerCfg",
    },
)

gym.register(
    id="Kevin-Mulag-HydraulicROM-JointTracking-v0",
    entry_point="kevin_integration.tasks.mulag_adaptive_rl_env:MulagAdaptiveRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagHydraulicROMJointTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagHydraulicROMJointTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Kevin-Mulag-UpperLevel-v0",
    entry_point="kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env:MulagUpperLevelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg:MulagUpperLevelEnvCfg",
        "rsl_rl_cfg_entry_point": "kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg:MulagUpperLevelPPORunnerCfg",
    },
)

# Clean post-PPO1 branch: standard task registration with ID/FD enabled and ROM disabled.
import ID_FD_pipline.tasks  # noqa: F401,E402
