# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass
from dataclasses import MISSING

import rrlab_tasks.manager_based.manipulation.obstacle_avoidance.mdp as mdp
from rrlab_tasks.manager_based.manipulation.obstacle_avoidance.obstacle_avoidance_env_cfg import ObstacleAvoidanceSceneCfg, ObstacleAvoidanceCfg
##
# Pre-defined configs
##
from rrlab_assets import MULAG_CFG  # isort: skip
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.sensors import TiledCameraCfg, FrameTransformerCfg, OffsetCfg, ContactSensorCfg
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from ...mdp.recorders_cfg import PostStepNormalizedJointPositionActionsRecorderCfg, PreStepStudentObservationsRecorderCfg, MulagStudentObsTeacherActRecorderManagerCfg


##
# Environment configuration
##

@configclass
class ObstacleAvoidanceSensorSceneCfg(ObstacleAvoidanceSceneCfg):
    contact_sensor_head = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Messerkopf",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        track_air_time=True,
    )

    contact_sensor_ausleger_2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Ausleger_II",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        track_air_time=True,
    )

    height_sensor_base: RayCasterCfg = RayCasterCfg( # height map
        prim_path="{ENV_REGEX_NS}/Robot/Base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(8.5, -2.7, 7),
        ),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.2,
            size=[8, 4]
        ),
        debug_vis=False,
    )


    height_sensor_left: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Messerkopf", 
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.4, -0.5, 0.5),
        ),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1,
            size=[0.05, 0.05]
        ),
        debug_vis=False,
    )


    height_sensor_right: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Messerkopf", 
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.4, 0.6, 0.5),
        ),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1,
            size=[0.05, 0.05]
        ),
        debug_vis=False,
    )

    distance_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Messerkopf", 
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.3), #rot=(0.707, 0, -0.7071068, 0)
        ),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, vertical_fov_range=[0, 0.1], horizontal_fov_range=[-90, 90], horizontal_res=5.0
        ),
        debug_vis=False,
    )

    tcp_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Base_link",
        target_frames=[FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Messerkopf",
            offset=OffsetCfg(pos=(0.4, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)))],
        debug_vis=False,
    )
    
    
    tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Messerkopf/tiled_camera",
        width=100,
        height=100,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.30, clipping_range=(0.1, 10.0) # focal_length= 9.0 for old model
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, -1.0), rot=(-0.3877, 0.59135, 0.59134, -0.38771), convention="usd"), # w,x,y,z
    )

@configclass
class MulagObstacleAvoidanceEnvCfg(ObstacleAvoidanceCfg):
    scene: ObstacleAvoidanceSensorSceneCfg = ObstacleAvoidanceSensorSceneCfg(num_envs=169, env_spacing=20)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.env_spacing = 20

        # switch robot to Mulag
        self.scene.robot = MULAG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (-25.4, 4.3, 0.0)

        # self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["Messerkopf"]
        # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["Messerkopf"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["Messerkopf"]
        # self.rewards.end_effector_orientation_tracking_fine_grained.params["asset_cfg"].body_names = ["Messerkopf"]

        # override actions
        # self.actions.arm_action = mdp.JointVelocityActionCfg(
        #     asset_name="robot",
        #     joint_names=[
        #         "Drehzapfen_joint",
        #         "Ausleger_I_joint",
        #         "Ausleger_II_joint",
        #         "Messerkopf_Schwenk_joint",
        #         "Messerkopf_joint"
        #         ],
        #     clip={"Drehzapfen_joint": (-2.0, 2.0),
        #           "Ausleger_I_joint": (-2.0, 2.0),
        #           "Ausleger_II_joint": (-2.0, 2.0),
        #           "Messerkopf_Schwenk_joint": (-2.0, 2.0),
        #           "Messerkopf_joint": (-2.0, 2.0),
        #           },
        #     use_default_offset=True,
        #     # scale=0.005,
        # )
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "Drehzapfen_joint",
                "Ausleger_I_joint",
                "Ausleger_II_joint",
                "Messerkopf_Schwenk_joint",
                "Messerkopf_joint"
                ],
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "Messerkopf"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


        # Action
        # self.actions.joint_pos = JointPositionNormalizedToLimitsActionCfg(asset_name="robot", joint_names=[".*"], scale=0.15)

        # # Recorder
        # self.recorders = MulagStudentObsTeacherActRecorderManagerCfg()
        # # self.recorders.record_pre_step_actions = PostStepNormalizedJointPositionActionsRecorderCfg()
        # # self.recorders.record_pre_step_observations = PreStepStudentObservationsRecorderCfg()
        # self.recorders.dataset_export_dir_path = "datasets"
        # self.recorders.dataset_filename = "mulag_teacher_sb3_lstm_ppo_dataset"


@configclass
class MulagObstacleAvoidanceEnvCfg_PLAY(MulagObstacleAvoidanceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.num_envs = 10
        # disable randomization for play
        self.observations.policy.enable_corruption = False
