# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# """Configuration for the Franka Emika robots.

# The following configurations are available:

# * :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
# * :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

# Reference: https://github.com/frankaemika/franka_ros
# """

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from rrlab_assets import RRLAB_ASSETS_DATA_DIR

##
# Configuration
##

SAUGBAGGER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{RRLAB_ASSETS_DATA_DIR}/Robots/Saugbagger_Arm_Phobos/Saugbagger_Arm_Phobos.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Link_0_link_joint": 1.0,
            "Link_1_link_joint": 0.569,
            "Link_2_link_joint": 1.5,
            "Link_3_link_joint": 0.0,
            "Link_4_link_joint": 0.7,
            "MTS_SteelPipe_link_joint": 0.0,
            "TCP_link_joint": 0.0,
        },
    ),
    actuators={
        "saugbagger_main_joints": ImplicitActuatorCfg(
            joint_names_expr=["Link_[0-4]_link_joint"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=5000.0,
            damping=50.0,
        ),
        # "saugbagger_pipe_joint": ImplicitActuatorCfg(
        #     joint_names_expr=["MTS_SteelPipe_link_joint"],
        #     effort_limit=12.0,
        #     velocity_limit=2.61,
        #     stiffness=80.0,
        #     damping=4.0,
        # ),
        # "saugbagger_TCP": ImplicitActuatorCfg(
        #     joint_names_expr=["TCP_link_joint"],
        #     effort_limit=200.0,
        #     velocity_limit=0.2,
        #     stiffness=2e3,
        #     damping=1e2,
        # ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


# FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
# FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
# """Configuration of Franka Emika Panda robot with stiffer PD control.

# This configuration is useful for task-space control using differential IK.
# """