# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    compute_pose_error,
    quat_from_euler_xyz,
    quat_unique,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import CustomUniformPoseCommandCfg


class CustomUniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands in the Sub-Environment coordinate system.

    Logic:
    1. **Initialization/Resample:** - Sample a "Lateral Offset" (Y), "Height" (Z), and "Orientation" (Roll/Pitch/Yaw) within the provided ranges.
       - These values are defined relative to the Sub-Environment Origin (0,0,0 of the specific env instance).
       - These values remain FIXED until the next reset/resample event.

    2. **Step Update:**
       - Get the Unimog's current position relative to the Sub-Environment Origin.
       - Extract the Unimog's X position.
       - Construct the Target Pose: 
         * Target X = Unimog X + (Optional X Offset)
         * Target Y = Fixed Y + Sampled Y
         * Target Z = Fixed Z + Sampled Z
         * Target Rot = Fixed Rot + Sampled Rot
       - Convert this Target Pose (which is in Sub-Env frame) into the Robot's Body Frame for visualization.
    """

    cfg: CustomUniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: CustomUniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # -- Buffers for Policy Input (Robot Base Frame)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        
        # -- Buffers for Metric Calculation (World Frame)
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # -- Storage for the "Environment Fixed" Parameters
        # These are sampled once per reset and stay fixed.
        # Format: [X_offset, Y_fixed, Z_fixed] relative to Sub-Env Origin
        self.env_fixed_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # Format: [qw, qx, qy, qz] relative to Sub-Env Origin
        self.env_fixed_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.env_fixed_quat[:, 0] = 1.0

        # -- Buffer for the computed target in World Frame (for Vis)
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand (Sub-Env Frame, X-Tracking):\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command in the robot base frame. Shape is (num_envs, 7)."""
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # Update metric buffers using the calculated world target
        self.pose_command_w[:, :3] = self.target_pos_w
        self.pose_command_w[:, 3:] = self.target_quat_w

        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """
        Samples the Y, Z, and Orientation components relative to the Sub-Env Origin.
        These act as the "lane" parameters that do not change until the next reset.
        """
        len_ids = len(env_ids)
        r = torch.empty(len_ids, device=self.device)

        # Sample Fixed Components in Sub-Env Frame
        # X is usually a small offset (or 0.0)
        self.env_fixed_pos[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        # Y is the lateral position in the "lane"
        self.env_fixed_pos[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        # Z is the height (not used later on)
        self.env_fixed_pos[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        # Sample Fixed Orientation in Sub-Env Frame
        euler_angles = torch.zeros((len_ids, 3), device=self.device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        if self.cfg.make_quat_unique:
            quat = quat_unique(quat)
            
        self.env_fixed_quat[env_ids] = quat

    def _update_command(self):
        """
        Called every step.
        1. Calculate Robot Position in Sub-Env Frame.
        2. Construct Target (Anchor) in Sub-Env Frame (Robot X + Fixed Y/Z).
        3. Convert Target Sub-Env -> World -> Robot Base Frame.
        """
        # --- 1. Get Robot State in World Frame ---
        robot_pos_w = self.robot.data.root_pos_w
        robot_quat_w = self.robot.data.root_quat_w

        # --- 2. Get Sub-Env Origins ---
        # The env_origins attribute contains the world position of (0,0,0) for each sub-env.
        env_origins = self._env.scene.env_origins 

        # --- 3. Compute Robot X in Sub-Env Frame ---
        # Since env_origins are usually just translations, we can subtract.
        robot_pos_env = robot_pos_w - env_origins
        robot_x_env = robot_pos_env[:, 0]

        # --- 4. Construct Target in Sub-Env Frame ---
        # Target X = Robot Current X (in env) + Sampled X Offset
        target_pos_env = self.env_fixed_pos.clone()
        target_pos_env[:, 0] = robot_x_env + self.env_fixed_pos[:, 0]
        
        # Target Rotation is the fixed sampled rotation
        target_quat_env = self.env_fixed_quat

        # --- 5. Convert Target Sub-Env -> World Frame ---
        # World = SubEnv_Origin + Target_SubEnv
        self.target_pos_w = target_pos_env + env_origins
        self.target_quat_w = target_quat_env # Assuming env origins have identity rotation

        # --- 6. Convert Target World -> Robot Base Frame (for Policy) ---
        # Command = inv(Robot_Pose) * Target_World
        self.pose_command_b[:, :3], self.pose_command_b[:, 3:] = subtract_frame_transforms(
            robot_pos_w, 
            robot_quat_w, 
            self.target_pos_w, 
            self.target_quat_w
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        
        # Visualize the computed World Target
        self.goal_pose_visualizer.visualize(self.target_pos_w, self.target_quat_w)
        #print("Target (pos): ", self.target_pos_w)
        #print("Target (or): ", self.target_quat_w)
        
        # Visualize Body Pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])