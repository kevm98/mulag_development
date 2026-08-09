# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from rrlab_assets import RRLAB_ASSETS_DATA_DIR

MULAG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{RRLAB_ASSETS_DATA_DIR}/Robots/Unimog_Mulag_FME500/Unimog_Mulag_FME500.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-25.4, 4.3, 1),  
        rot=(1.0, 0.0, 0.0, 0.0), 
        joint_pos={ # radians
            "Drehzapfen_joint": 0.1, 
            "Ausleger_I_joint":  -0.02, 
            "Ausleger_II_joint": 0.0, 
            "Messerkopf_Schwenk_joint": 0.0, 
            "Messerkopf_joint": 0.0, 
            "Wheel_Front_Left_Steering_joint": 0.0, 
            "Wheel_Front_Right_Steering_joint": 0.0, 
            "Wheel_Front_Left_joint": 0.0, 
            "Wheel_Front_Right_joint": 0.0, 
            "Wheel_Rear_Left_joint": 0.0, 
            "Wheel_Rear_Right_joint": 0.0
        },
    ),
    actuators={
        "mulag_arm_1": ImplicitActuatorCfg(
            joint_names_expr=[
                "Drehzapfen_joint",
            ],
            effort_limit=100000.0,
	        velocity_limit_sim=1.0,
            stiffness=30000.0, # if stiff >>> damp, mulag starts flying, still needs tuning
            damping=30000.0,   
        ),
        "mulag_arm_2": ImplicitActuatorCfg(
            joint_names_expr=[
                "Ausleger_I_joint",
            ],
            effort_limit=100000.0,
	        velocity_limit_sim=1.0,
            stiffness=10000.0, # if stiff >>> damp, mulag starts flying, still needs tuning
            damping=10000.0,   
        ),
        "mulag_arm_3": ImplicitActuatorCfg(
            joint_names_expr=[
                "Ausleger_II_joint",
            ],
            effort_limit=100000.0,
	        velocity_limit_sim=1.0,
            stiffness=10000.0, # if stiff >>> damp, mulag starts flying, still needs tuning
            damping=10000.0,   
        ),
        "mulag_arm_4": ImplicitActuatorCfg(
            joint_names_expr=[
                "Messerkopf_Schwenk_joint",
            ],
            effort_limit=100000.0,
	        velocity_limit_sim=1.0,
            stiffness=6000.0, # if stiff >>> damp, mulag starts flying, still needs tuning
            damping=6000.0,   
        ),
        "mulag_arm_5": ImplicitActuatorCfg(
            joint_names_expr=[
                "Messerkopf_joint"
            ],
            effort_limit=100000.0,
	        velocity_limit_sim=1.0,
            stiffness=6000.0, # if stiff >>> damp, mulag starts flying, still needs tuning
            damping=6000.0,   
        ),
        "unimog_wheels_drive": ImplicitActuatorCfg(
            joint_names_expr=["Wheel_Front_Left_joint", "Wheel_Front_Right_joint", "Wheel_Rear_Left_joint", "Wheel_Rear_Right_joint"],
            stiffness=600000000.0,
            damping=2000000.0,  
        ),
        "unimog_steering": ImplicitActuatorCfg(
            joint_names_expr=["Wheel_Front_Left_Steering_joint", "Wheel_Front_Right_Steering_joint"],
            stiffness=10.0,
            damping=1000000000.0,  
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
