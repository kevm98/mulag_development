from __future__ import annotations

import gymnasium as gym


gym.register(
    id="Kevin-Mulag-AdaptiveRL-JointReach-v0",
    entry_point="kevin_integration.tasks.mulag_adaptive_rl_env:MulagAdaptiveRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagAdaptiveRLJointReachEnvCfg",
        "rsl_rl_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagAdaptiveRLPPORunnerCfg",
    },
)

gym.register(
    id="Kevin-Mulag-AdaptiveRL-v0",
    entry_point="kevin_integration.tasks.mulag_adaptive_rl_env:MulagAdaptiveRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagAdaptiveRLJointReachEnvCfg",
        "rsl_rl_cfg_entry_point": "kevin_integration.tasks.mulag_adaptive_rl_env_cfg:MulagAdaptiveRLPPORunnerCfg",
    },
)
