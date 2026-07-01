# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, joint_pos_env_cfg
import os
##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="RRLAB-Obstacle-Avoidance-Mulag-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.MulagObstacleAvoidanceEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MulagObstacleAvoidancePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_contrib_cfg_entry_point": f"{agents.__name__}:sb3_ppo_lstm_cfg.yaml",
        "imitation_cfg_entry_point": f"{agents.__name__}:imitation_dagger_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_depth.json"),
        "robomimic_bc_rnn_gmm_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_gmm_depth.json"),
    },
)
