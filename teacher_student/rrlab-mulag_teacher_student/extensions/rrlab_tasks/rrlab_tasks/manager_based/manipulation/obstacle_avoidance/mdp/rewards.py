# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply_inverse, transform_points, euler_xyz_from_quat
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    
def lateral_impact_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Penalizes forces in the X and Y directions (collisions), ignores Z (support weight).
    """

    contact_sensor = env.scene[sensor_cfg.name]
    tcp_data = env.scene["tcp_transformer"].data

    forces = contact_sensor.data.net_forces_w # [N, 1, 3]
    #print("Forces: ", forces)

    rotated_forces = quat_apply_inverse(tcp_data.target_quat_w, forces)
    #print("rotated force: ", rotated_forces)
    
    lateral_forces = rotated_forces[..., :2] # only focus on force in x and y direction

    impact_magnitude = torch.norm(lateral_forces, dim=-1)

    max_impact = torch.max(impact_magnitude, dim=1)[0] # if more than one contact sensor
    #print("Impact: ", max_impact)

    return torch.where(max_impact > threshold, -1.0, 0.0)



def contact_detection(env: ManagerBasedRLEnv, min_contact_time: float = 0.07) -> torch.Tensor:
    """
    Penalizes contact time of Contact Sensor based on contact time
    """
    data_contact_sensor = env.scene["contact_sensor_head"].data
    contact_time = data_contact_sensor.current_contact_time.flatten()

    # print("Contact Time: ", contact_time) 

    reward = torch.tanh(contact_time - min_contact_time)
    #print("Reward: ", reward)
    return reward
    


def tcp_height_to_terrain(env: ManagerBasedRLEnv, sensor_one_cfg: SceneEntityCfg, sensor_two_cfg: SceneEntityCfg, target_height: float = 0.02) -> torch.Tensor:
    """Penalize uneven orientation to ground slope using the ray caster and frame transformer

    The function computes the height difference of the TCP towards to terrain -offset. The rays in world coordinates are transformed into the 
    local TCP Coordinate Frame using the TCP Frame Transformer. Subsequently the mean of the distance of the TCP to the terrain is normalized (tanh) 
    and returned as the reward (penality)
    """
    left_data = env.scene[sensor_one_cfg.name].data
    right_data = env.scene[sensor_two_cfg.name].data
    tcp_data = env.scene["tcp_transformer"].data
    

    hits_left = left_data.ray_hits_w # [#Sensors/envs, #Rays, 3] in world coordinates
    hits_right = right_data.ray_hits_w # [#Sensors/envs, #Rays, 3] in world coordinates

    # isaaclab.utils transform_points not working correctly?
    translated_hits_left = hits_left - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame
    transposed_hits_left = quat_apply_inverse(tcp_data.target_quat_w, translated_hits_left) # rotate coordinate of casted ray of world coordinate into local coordinate frame
    
    
    height_of_ray_left = transposed_hits_left[:, :, 2] # only get height (z) of transposed rays
    
    
    translated_hits_right = hits_right - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame
    transposed_hits_right = quat_apply_inverse(tcp_data.target_quat_w, translated_hits_right) # rotate coordinate of casted ray of world coordinate into local coordinate frame


    height_of_ray_right = transposed_hits_right[:, :, 2] # only get height (z) of transposed rays

    avg_height = (height_of_ray_left + height_of_ray_right) / 2.0
    # print("avg_height shape: ", avg_height.shape)

    error = avg_height.squeeze(-1) - target_height
    # print("reward height error shape: ", error.shape)
    
    # print("Reward (height_tcp): ", error)
    return error    

def tcp_height_to_terrain_tanh(env: ManagerBasedRLEnv, sensor_one_cfg: SceneEntityCfg, sensor_two_cfg: SceneEntityCfg, target_height: float = 0.02) -> torch.Tensor:
    """Penalize uneven orientation to ground slope using the ray caster and frame transformer

    The function computes the height difference of the TCP towards to terrain -offset. The rays in world coordinates are transformed into the 
    local TCP Coordinate Frame using the TCP Frame Transformer. Subsequently the mean of the distance of the TCP to the terrain is normalized (tanh) 
    and returned as the reward (penality)
    """
    left_data = env.scene[sensor_one_cfg.name].data
    right_data = env.scene[sensor_two_cfg.name].data
    tcp_data = env.scene["tcp_transformer"].data
    

    hits_left = left_data.ray_hits_w # [#Sensors/envs, #Rays, 3] in world coordinates
    hits_right = right_data.ray_hits_w # [#Sensors/envs, #Rays, 3] in world coordinates

    # isaaclab.utils transform_points not working correctly?
    translated_hits_left = hits_left - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame
    transposed_hits_left = quat_apply_inverse(tcp_data.target_quat_w, translated_hits_left) # rotate coordinate of casted ray of world coordinate into local coordinate frame
    
    
    height_of_ray_left = transposed_hits_left[:, :, 2] # only get height (z) of transposed rays
    
    
    translated_hits_right = hits_right - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame
    transposed_hits_right = quat_apply_inverse(tcp_data.target_quat_w, translated_hits_right) # rotate coordinate of casted ray of world coordinate into local coordinate frame


    height_of_ray_right = transposed_hits_right[:, :, 2] # only get height (z) of transposed rays

    avg_height = (height_of_ray_left + height_of_ray_right) / 2.0
    #print("Avg: ", avg_height)

    # error = avg_height.flatten() - target_height

    # sigma = 0.07 # sigma is tolerance
    # tmp = torch.exp(- (error**2) / (2 * sigma**2))
    
    # print("Reward (height_tcp): ", tmp)
    return 1 - torch.tanh(avg_height.flatten()/target_height)

def tcp_height_to_terrain_abs(env: ManagerBasedRLEnv, sensor_one_cfg: SceneEntityCfg, sensor_two_cfg: SceneEntityCfg, threshold: float = 0.1) -> torch.Tensor:

    left_data = env.scene[sensor_one_cfg.name].data
    right_data = env.scene[sensor_two_cfg.name].data
    tcp_data = env.scene["tcp_transformer"].data
    

    hits_left = left_data.ray_hits_w # [#Sensors/envs, #Rays, 3] in world coordinates
    hits_right = right_data.ray_hits_w # [#Sensors/envs, #Rays, 3] in world coordinates

    # isaaclab.utils transform_points not working correctly?
    translated_hits_left = hits_left - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame
    transposed_hits_left = quat_apply_inverse(tcp_data.target_quat_w, translated_hits_left) # rotate coordinate of casted ray of world coordinate into local coordinate frame
    height_of_ray_left = transposed_hits_left[:, :, 2] # only get height (z) of transposed rays
    
    
    translated_hits_right = hits_right - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame
    transposed_hits_right = quat_apply_inverse(tcp_data.target_quat_w, translated_hits_right) # rotate coordinate of casted ray of world coordinate into local coordinate frame


    height_of_ray_right = transposed_hits_right[:, :, 2] # only get height (z) of transposed rays

    delta_height = (height_of_ray_left - height_of_ray_right).squeeze()
    # print("abs: ", abs_height)
    
    return torch.abs(delta_height) - threshold

def tcp_height_to_terrain_abs_tanh(env: ManagerBasedRLEnv, sensor_one_cfg: SceneEntityCfg, sensor_two_cfg: SceneEntityCfg, std: float = 0.1) -> torch.Tensor:

    left_data = env.scene[sensor_one_cfg.name].data
    right_data = env.scene[sensor_two_cfg.name].data
    tcp_data = env.scene["tcp_transformer"].data
    

    hits_left = left_data.ray_hits_w # [#Sensors/envs, #Rays, 3] in world coordinates
    hits_right = right_data.ray_hits_w # [#Sensors/envs, #Rays, 3] in world coordinates

    # isaaclab.utils transform_points not working correctly?
    translated_hits_left = hits_left - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame
    transposed_hits_left = quat_apply_inverse(tcp_data.target_quat_w, translated_hits_left) # rotate coordinate of casted ray of world coordinate into local coordinate frame
    height_of_ray_left = transposed_hits_left[:, :, 2] # only get height (z) of transposed rays
    
    
    translated_hits_right = hits_right - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame
    transposed_hits_right = quat_apply_inverse(tcp_data.target_quat_w, translated_hits_right) # rotate coordinate of casted ray of world coordinate into local coordinate frame


    height_of_ray_right = transposed_hits_right[:, :, 2] # only get height (z) of transposed rays

    delta_height = (height_of_ray_left - height_of_ray_right).squeeze()
    # print("abs: ", abs_height)
    
    return 1 - torch.tanh(torch.abs(delta_height)/std) # ca. 0.05 minimum



def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, offset: float = 0.4) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm (y-direction).

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_link_state_w[:, :3], asset.data.root_link_state_w[:, 3:7], des_pos_b)
    reduced_des_pos_w = des_pos_w[:, :2] # only want to penalize deviation from mowing distance in xy


    curr_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    reduced_curr_pos_w = curr_pos_w[:, :2] # only want to penalize deviation from mowing distance in xy

    reward = torch.norm(reduced_curr_pos_w - reduced_des_pos_w, dim=1) 
    #print("Reward (pos): ", reward)
    return reward - offset



def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel (y-direction).

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_link_state_w[:, :3], asset.data.root_link_state_w[:, 3:7], des_pos_b)
    reduced_des_pos_w = des_pos_w[:, :2] # only want to reward moving in xy direction


    curr_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    reduced_curr_pos_w = curr_pos_w[:, :2] # only want to reward moving in xy direction

    distance = torch.norm(reduced_curr_pos_w - reduced_des_pos_w, dim=1)  # if pos is more than 1 dimensional
    # print("reward distance: ", distance)
    reward = 1 - torch.tanh(distance / std)
    #print("Reward (pos tanh): ", reward)
    return reward


# def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking orientation error using shortest path.

#     The function computes the orientation error between the desired orientation (from the command) and the
#     current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
#     path between the desired and current orientations.
#     """
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)

#     des_quat_b = command[:, 3:7]
#     des_quat_w = quat_mul(asset.data.root_link_state_w[:, 3:7], des_quat_b)
#     curr_quat_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

#     reward = quat_error_magnitude(curr_quat_w, des_quat_w)
#     print("Reward (orientation): ", reward)
#     return reward




def yaw_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, offset_deg: float = 30) -> torch.Tensor:
    """Yaw-only tracking error."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # desired quat in body frame
    des_quat_b = command[:, 3:7]

    # convert to world frame
    des_quat_w = quat_mul(asset.data.root_link_state_w[:, 3:7], des_quat_b)

    # current orientation
    curr_quat_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], 3:7]

    # --- extract yaw ---
    _, _, des_yaw = euler_xyz_from_quat(des_quat_w)
    _, _, curr_yaw = euler_xyz_from_quat(curr_quat_w)

    # shortest yaw difference
    yaw_error = torch.atan2(
        torch.sin(curr_yaw - des_yaw),
        torch.cos(curr_yaw - des_yaw),
    )

    # return absolute error or squared
    return torch.abs(yaw_error) - offset_deg * (torch.pi / 180)

def pitch_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, offset_deg: float = 30) -> torch.Tensor:
    """pitch-only tracking error."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # desired quat in body frame
    des_quat_b = command[:, 3:7]

    # convert to world frame
    des_quat_w = quat_mul(asset.data.root_link_state_w[:, 3:7], des_quat_b)

    # current orientation
    curr_quat_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], 3:7]

    # --- extract yaw ---
    _, des_pitch, _ = euler_xyz_from_quat(des_quat_w)
    _, curr_pitch, _ = euler_xyz_from_quat(curr_quat_w)

    # shortest yaw difference
    pitch_error = torch.atan2(
        torch.sin(curr_pitch - des_pitch),
        torch.cos(curr_pitch - des_pitch),
    )

    # return absolute error or squared
    return torch.abs(pitch_error) - offset_deg * (torch.pi / 180)

# def orientation_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking orientation error using shortest path.

#     The function computes the orientation error between the desired orientation (from the command) and the
#     current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
#     path between the desired and current orientations.
#     """
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)

#     des_quat_b = command[:, 3:7]
#     des_quat_w = quat_mul(asset.data.root_link_state_w[:, 3:7], des_quat_b)
#     curr_quat_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

#     distance = quat_error_magnitude(curr_quat_w, des_quat_w)
#     return (1 - torch.tanh(distance / std))


def yaw_command_error_tanh(env: ManagerBasedRLEnv, std_deg: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Yaw-only tracking error."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # desired quat in body frame
    des_quat_b = command[:, 3:7]

    # convert to world frame
    des_quat_w = quat_mul(asset.data.root_link_state_w[:, 3:7], des_quat_b)

    # current orientation
    curr_quat_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], 3:7]

    # --- extract yaw ---
    _, _, des_yaw = euler_xyz_from_quat(des_quat_w)
    _, _, curr_yaw = euler_xyz_from_quat(curr_quat_w)

    # shortest yaw difference
    yaw_error = torch.atan2(
        torch.sin(curr_yaw - des_yaw),
        torch.cos(curr_yaw - des_yaw),
    )

    # return absolute error or squared
    return (1 - torch.tanh(torch.abs(yaw_error) / (std_deg * (torch.pi / 180))))

def pitch_command_error_tanh(env: ManagerBasedRLEnv, std_deg: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """pitch-only tracking error."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # desired quat in body frame
    des_quat_b = command[:, 3:7]

    # convert to world frame
    des_quat_w = quat_mul(asset.data.root_link_state_w[:, 3:7], des_quat_b)

    # current orientation
    curr_quat_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], 3:7]

    # --- extract yaw ---
    _, des_pitch, _ = euler_xyz_from_quat(des_quat_w)
    _, curr_pitch, _ = euler_xyz_from_quat(curr_quat_w)

    # shortest yaw difference
    pitch_error = torch.atan2(
        torch.sin(curr_pitch - des_pitch),
        torch.cos(curr_pitch - des_pitch),
    )

    # return absolute error or squared
    return (1 - torch.tanh(torch.abs(pitch_error) / (std_deg * (torch.pi / 180))))


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, offset_deg: float = 30) -> torch.Tensor:
    """Yaw-only tracking error."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # desired quat in body frame
    des_quat_b = command[:, 3:7]

    # convert to world frame
    des_quat_w = quat_mul(asset.data.root_link_state_w[:, 3:7], des_quat_b)

    # current orientation
    curr_quat_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], 3:7]

    # --- extract yaw ---
    _, _, des_yaw = euler_xyz_from_quat(des_quat_w)
    _, _, curr_yaw = euler_xyz_from_quat(curr_quat_w)

    # shortest yaw difference
    yaw_error = torch.atan2(
        torch.sin(curr_yaw - des_yaw),
        torch.cos(curr_yaw - des_yaw),
    )

    # return absolute error or squared
    return torch.abs(yaw_error) - offset_deg * (torch.pi / 180)

def orientation_command_error_tanh(env: ManagerBasedRLEnv, std_deg: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Yaw-only tracking error."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # desired quat in body frame
    des_quat_b = command[:, 3:7]

    # convert to world frame
    des_quat_w = quat_mul(asset.data.root_link_state_w[:, 3:7], des_quat_b)

    # current orientation
    curr_quat_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], 3:7]

    # --- extract yaw ---
    _, _, des_yaw = euler_xyz_from_quat(des_quat_w)
    _, _, curr_yaw = euler_xyz_from_quat(curr_quat_w)

    # shortest yaw difference
    yaw_error = torch.atan2(
        torch.sin(curr_yaw - des_yaw),
        torch.cos(curr_yaw - des_yaw),
    )

    # return absolute error or squared
    return (1 - torch.tanh(torch.abs(yaw_error) / (std_deg * (torch.pi / 180))))


def tcp_ahead_of_arm_base(env: ManagerBasedRLEnv, tcp_cfg: SceneEntityCfg, arm_base_cfg: SceneEntityCfg, threshold: float = 0.2) -> torch.Tensor:
    """Checks whether the TCP is ahead of the Unimog."""
    tcp = env.scene[tcp_cfg.name]
    arm_base = env.scene[arm_base_cfg.name]

    tcp_pos_w = tcp.data.body_link_state_w[:, tcp_cfg.body_ids[0], :3]
    arm_base_pos_w = arm_base.data.body_link_state_w[:, arm_base_cfg.body_ids[0], :3]

    distance_x = tcp_pos_w[:, 0] - arm_base_pos_w[:, 0]
    # print("TCP ahead distance: ", distance_x)

    return torch.where(distance_x > threshold, 10.0, distance_x - threshold)

class tcp_hovering_above_terrain(ManagerTermBase):
    def __init__(self, cfg: SceneEntityCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.timer = torch.zeros(env.num_envs, device=env.device)
        self.above_state = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, sensor_one_cfg: SceneEntityCfg, sensor_two_cfg: SceneEntityCfg, 
                 height_threshold: float = 0.4, hysteresis: float = 0.1, time_threshold_sec: float = 0.5) -> torch.Tensor:
        left_data = env.scene[sensor_one_cfg.name].data
        right_data = env.scene[sensor_two_cfg.name].data
        tcp_data = env.scene["tcp_transformer"].data
        dt = env.step_dt

        hits_left = left_data.ray_hits_w # [#Sensors/envs, #Rays, 3] in world coordinates
        hits_right = right_data.ray_hits_w # [#Sensors/envs, #Rays, 3] in world coordinates

        # isaaclab.utils transform_points not working correctly?
        translated_hits_left = hits_left - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame
        transposed_hits_left = quat_apply_inverse(tcp_data.target_quat_w, translated_hits_left) # rotate coordinate of casted ray of world coordinate into local coordinate frame
        
        
        height_of_ray_left = transposed_hits_left[:, :, 2] # only get height (z) of transposed rays
        
        
        translated_hits_right = hits_right - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame
        transposed_hits_right = quat_apply_inverse(tcp_data.target_quat_w, translated_hits_right) # rotate coordinate of casted ray of world coordinate into local coordinate frame


        height_of_ray_right = transposed_hits_right[:, :, 2] # only get height (z) of transposed rays

        avg_height = ((height_of_ray_left + height_of_ray_right) / 2.0).squeeze(-1)
        # print("hovering Avg: ", avg_height)

        enter = avg_height > (height_threshold + hysteresis)      # e.g. +0.01
        exit_  = avg_height < (height_threshold - hysteresis)     # e.g. -0.01

        self.above_state = torch.where(self.above_state, ~exit_, enter)

        self.timer += self.above_state.float() * dt
        self.timer *= self.above_state
        # print("reward hovering timer: ", self.timer)
        
        # if tcp is above terrain for more than time threshold, return 1, else return 0
        return torch.where(self.timer > time_threshold_sec, 1.0, 0.0)
    
    def reset(self, env_ids=None):
        if env_ids is None:
            self.timer[:] = 0.0
            self.above_state[:] = False
        else:
            self.timer[env_ids] = 0.0
            self.above_state[env_ids] = False


class tcp_off_working_space(ManagerTermBase):
    def __init__(self, cfg: SceneEntityCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.timer = torch.zeros(env.num_envs, device=env.device)
        self.above_state = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float, hysteresis: float, time_threshold_sec: float):
        robot = env.scene[asset_cfg.name]
        dt = env.step_dt

        tcp_pos_w = robot.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]
        base_pos_w = robot.data.root_pos_w
        base_rotations = robot.data.root_quat_w

        tcp_pos_base = tcp_pos_w - base_pos_w
        tcp_pos_base_y = quat_apply_inverse(base_rotations, tcp_pos_base)[:, 1].squeeze(-1)
        # print("reward tcp_pos_base_y: ", tcp_pos_base_y)

        enter = tcp_pos_base_y > (threshold + hysteresis)      # e.g. +0.01
        exit_  = tcp_pos_base_y < (threshold - hysteresis)     # e.g. -0.01

        self.above_state = torch.where(self.above_state, ~exit_, enter)

        self.timer += self.above_state.float() * dt
        self.timer *= self.above_state

        # print("reward off timer: ", self.timer)
        
        # if tcp is above terrain for more than time threshold, return 1, else return 0
        return torch.where(self.timer > time_threshold_sec, 1.0, 0.0)
    
    def reset(self, env_ids=None):
        if env_ids is None:
            self.timer[:] = 0.0
            self.above_state[:] = False
        else:
            self.timer[env_ids] = 0.0
            self.above_state[env_ids] = False

def tcp_off_working_space_tanh(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float, std: float):
    robot = env.scene[asset_cfg.name]

    tcp_pos_w = robot.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]
    base_pos_w = robot.data.root_pos_w
    base_rotations = robot.data.root_quat_w

    tcp_pos_base = tcp_pos_w - base_pos_w
    tcp_pos_base_y = quat_apply_inverse(base_rotations, tcp_pos_base)[:, 1].squeeze(-1)

    dist = tcp_pos_base_y - threshold
    # print("reward dist: ", dist)
    return torch.where(dist >= 0, torch.tanh(dist/std), 0)
    


class finish_on_limited_contacts(ManagerTermBase):
    """Dense reward that prefers: higher success/progress + fewer contacts.

    Reward structure:
      progress ∈ [0, 1]
      contact_score ∈ [0, 1]  (1 = no contacts, 0 = too many contacts)
      impact_score ∈ [0, 1]   (1 = no impact, 0 = large impact)

      base_reward = progress * contact_score * impact_score
      finish_bonus = finished * finish_bonus_weight * contact_score

    Contacts are counted as *events* using hysteresis, so sustained scraping is not
    counted every step.
    """

    def __init__(self, cfg: SceneEntityCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        device = env.device
        n = env.num_envs
        self.contacts_counter = torch.zeros(n, device=device, dtype=torch.int32)
        self._in_contact = torch.zeros(n, device=device, dtype=torch.bool)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        traj_length_x: float,
        contact_force_threshold: float,
        max_contacts_count: int,
        # track / termination
        # off_track_y: float = 16.0,          # absolute Y deviation threshold (adjust for your track)
        # contact event hysteresis
        release_ratio: float = 0.5,         # release when force < release_ratio * threshold
        # impact shaping
        impact_beta: float = 0.2,           # how much impact reduces contact_score (0..1)
        impact_cap: float = 10.0,           # cap scaled impact before tanh (safety)
        # finish bonus
        finish_bonus_weight: float = 2.0,   # bonus magnitude at finish
    ) -> torch.Tensor:
        # -----------------------------
        # Robot state
        # -----------------------------
        robot = env.scene[asset_cfg.name]
        env_origins = env.scene.env_origins              # [N, 3]
        root_pos_w = robot.data.root_pos_w               # [N, 3]

        # -----------------------------
        # Progress (success rate proxy): along +X
        # -----------------------------
        dx = (root_pos_w[:, 0] - env_origins[:, 0] + 25) # robot start at -25 in x axis of local env, local env length set to 45 in terrain.
        # print("reward finish on limited contacts, dx: ", dx)
        progress = dx / max(traj_length_x, 1e-6)
        progress = torch.clamp(progress, 0.0, 1.0)       # [N]
        finished = progress >= 1.0                       # [N] bool
        # print("reward, progress: ", progress)

    # lateral_deviation = (env_origins[:, 0] - root_positions[:, 0]) 
    # print("DEV: ", lateral_deviation)

    # is_off_track = lateral_deviation < -16.0
        # # -----------------------------
        # # Off-track (optional)
        # # -----------------------------
        # # If "lateral" is Y in your setup, use Y:
        # lateral_dev = (root_pos_w[:, 1] - env_origins[:, 1])
        # is_off_track = torch.abs(lateral_dev) > off_track_y

        # -----------------------------
        # Contact / impact (TCP frame)
        # -----------------------------
        contact_sensor = env.scene[sensor_cfg.name]
        tcp_data = env.scene["tcp_transformer"].data

        forces_w = contact_sensor.data.net_forces_w      # [N, M, 3]
        # rotate forces into TCP frame (so "lateral" means in-plane)
        forces_tcp = quat_apply_inverse(tcp_data.target_quat_w, forces_w)

        lateral_forces = forces_tcp[..., :2]             # XY only
        impact_mag = torch.norm(lateral_forces, dim=-1)  # [N, M]
        max_impact = torch.max(impact_mag, dim=1).values # [N]
        # print("reward finish on limited contacts, max_impact: ", max_impact)
        # -----------------------------
        # Count contact *events* with hysteresis
        # -----------------------------
        thr = max(contact_force_threshold, 1e-6)
        start_event = (max_impact > thr) & (~self._in_contact)
        end_event = (max_impact < (release_ratio * thr)) & (self._in_contact)

        self.contacts_counter += start_event.to(torch.int32)
        # update in_contact state
        self._in_contact = torch.where(end_event, torch.zeros_like(self._in_contact), self._in_contact)
        self._in_contact = torch.where(start_event, torch.ones_like(self._in_contact), self._in_contact)

        # -----------------------------
        # Contact score in [0,1] (fewer contacts => higher)
        # -----------------------------
        max_c = max(int(max_contacts_count), 1)
        contact_frac = torch.clamp(self.contacts_counter.float() / float(max_c), 0.0, 1.0)
        contact_score = 1.0 - contact_frac               # 1 (no contacts) -> 0 (too many)
        # print("reward finish on limited contacts, contact_score: ", contact_score)
        # print("reward finish on limited contacts, self.contacts_counter: ", self.contacts_counter)
        # -----------------------------
        # Impact score in [0,1] (smaller impact => higher)
        # -----------------------------
        # scaled impact then tanh gives (0..1). Convert to score = 1 - penalty.
        scaled = torch.clamp(max_impact / thr, 0.0, impact_cap)
        impact_pen = torch.tanh(scaled)                  # 0..1
        impact_score = torch.clamp(1.0 - impact_beta * impact_pen, 0.0, 1.0)

        # Combine contact quality
        contact_quality = torch.clamp(contact_score * impact_score, 0.0, 1.0)
        # print("reward finish on limited contacts, contact_quality: ", contact_quality)
        # -----------------------------
        # Final reward: progress gated by contact quality
        # -----------------------------
        # print("reward, progress: ", progress, ", contact_quality: ", contact_quality, ", contact_score: ", contact_score)
        reward = 0.2 * progress + 0.4 * contact_quality + 0.4 * contact_score
        # print("reward finish on limited contacts, reward: ", reward)

        # -----------------------------
        # Bonus at finish (your requested part)
        # -----------------------------
        # Finishing cleanly yields more bonus than finishing with many contacts.
        finish_bonus = finished.to(torch.float32) * finish_bonus_weight * contact_quality
        # print("reward, finsh_bonus: ", finish_bonus)
        reward = reward + finish_bonus
        # print("reward finish on limited contacts, reward: ", reward)
        # Optional: punish off-track
        # reward = torch.where(is_off_track, reward - 1.0, reward)

        return reward

    def reset(self, env_ids=None):
        if env_ids is None:
            self.contacts_counter[:] = 0
            self._in_contact[:] = False
        else:
            self.contacts_counter[env_ids] = 0
            self._in_contact[env_ids] = False


def distance_to_obstacle(env: ManagerBasedRLEnv, safe_distance: float = 1.0, std: float = 1.0):
        distance_rays = env.scene["distance_scanner"].data.ray_hits_w
        tcp_data = env.scene["tcp_transformer"].data

        # isaaclab.utils transform_points not working correctly
        R = distance_rays.shape[1]
        translated_hits = distance_rays - tcp_data.target_pos_w # translate coordinate of casted ray of world coordinate into local coordinate frame

        transposed_hits = quat_apply_inverse(tcp_data.target_quat_w.expand(-1, R, -1), translated_hits) # rotate coordinate of casted ray of world coordinate into local coordinate frame
        invalid = torch.isnan(transposed_hits).any(dim=-1)

        distance = torch.norm(transposed_hits, dim=-1)                           # (N, R)
        distance = torch.where(invalid, torch.full_like(distance, 15), distance)

        distance = torch.clamp(distance , 0.0, 15.0)
        min_distance = distance.min(dim=1).values

        return 1 - torch.tanh((min_distance - safe_distance) / std)
