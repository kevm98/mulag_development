# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import RRLABUnitreeGo2RoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from .mdp.recorders_cfg import PostStepNormalizedJointPositionActionsRecorderCfg
from .mdp.actions_cfg import JointPositionNormalizedToLimitsActionCfg
@configclass
class RRLABUnitreeGo2FlatEnvCfg(RRLABUnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Action
        # self.actions.joint_pos = JointPositionNormalizedToLimitsActionCfg(asset_name="robot", joint_names=[".*"], scale=0.15)

        # Recorder
        # self.recorders = ActionStateRecorderManagerCfg()
        # self.recorders.record_pre_step_actions = PostStepNormalizedJointPositionActionsRecorderCfg()
        # self.recorders.dataset_export_dir_path = "/home/qili/Software/IsaacSim_Extensions/rrlab/datasets"
        # self.recorders.dataset_filename = "go2_flat_env_dataset_1"


class RRLABUnitreeGo2FlatEnvCfg_PLAY(RRLABUnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
