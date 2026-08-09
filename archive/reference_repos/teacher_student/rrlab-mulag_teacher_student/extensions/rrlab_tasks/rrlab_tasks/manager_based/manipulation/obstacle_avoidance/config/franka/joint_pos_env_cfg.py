# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import rrlab_tasks.manager_based.manipulation.obstacle_avoidance.mdp as mdp
from rrlab_tasks.manager_based.manipulation.obstacle_avoidance.obstacle_avoidance_env_cfg import ObstacleAvoidanceSceneCfg, ObstacleAvoidanceCfg
##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

##
# Environment configuration
##

@configclass
class ObstacleAvoidanceRGBCameraSceneCfg(ObstacleAvoidanceSceneCfg):

    # add camera to the scene
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
    )

@configclass
class FrankaObstacleAvoidanceEnvCfg(ObstacleAvoidanceCfg):
    scene: ObstacleAvoidanceRGBCameraSceneCfg = ObstacleAvoidanceRGBCameraSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class FrankaObstacleAvoidanceEnvCfg_PLAY(FrankaObstacleAvoidanceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
