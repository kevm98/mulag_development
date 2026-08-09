# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply_inverse, transform_points, euler_xyz_from_quat
from isaaclab.managers.manager_base import ManagerTermBase

def unimog_lateral_deviation(env: ManagerBasedRLEnv, asset_name: str="robot", deviation_threshold: float = 0.4) -> torch.Tensor:
    """
    Checks for all environments whether the Unimog deviates laterally from the straight path (X-axis).

    """
    robot: Articulation = env.scene[asset_name]

    env_origins = env.scene.env_origins # position of env origin in world coordinates

    root_positions = robot.data.root_pos_w # position of root link in world coordinates

    lateral_deviation = torch.abs((env_origins[:, 1] - root_positions[:, 1])) - 4.3 # get deviation in y direction between origin and root link, account for offset (4.2m)
    #print("DEV: ", lateral_deviation)

    is_off_track = torch.abs(lateral_deviation) > deviation_threshold
    return is_off_track


def unimog_end_of_lane(env: ManagerBasedRLEnv, asset_name: str="robot") -> torch.Tensor:
    """
    Checks for all environments whether the Unimog is at the end of its lane

    """
    robot: Articulation = env.scene[asset_name]

    env_origins = env.scene.env_origins # position of env origin in world coordinates

    root_positions = robot.data.root_pos_w # position of root link in world coordinates

    lateral_deviation = (env_origins[:, 0] - root_positions[:, 0]) 
    # print("DEV: ", lateral_deviation)

    is_off_track = lateral_deviation < -16.0
    # print("termination is off track: ", is_off_track)
    return is_off_track

def unimog_flying(env: ManagerBasedRLEnv, asset_name: str="robot", threshold: float = 0.05) -> torch.Tensor:
    """
    Detect if vehicle is lifted/tilted by arm (wheels off ground).
    Returns: (num_env,) bool tensor
    """

    robot: Articulation = env.scene[asset_name]

    base_quat = robot.data.root_quat_w  # (num_env, 4)

    # euler_xyz_from_quat returns (roll, pitch, yaw) as tuple of tensors
    roll, pitch, _ = euler_xyz_from_quat(base_quat)

    # (num_env, 2)
    roll_pitch = torch.stack((roll, pitch), dim=-1)

    # (num_env,)
    is_flying = torch.any(torch.abs(roll_pitch) > threshold, dim=-1)
    return is_flying

def impact_with_terrain(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 100000.0) -> torch.Tensor:
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

    return max_impact > threshold


def tcp_height_to_terrain_abs_difference(env: ManagerBasedRLEnv, sensor_one_cfg: SceneEntityCfg, sensor_two_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:

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
    
    return delta_height > threshold


def tcp_face_backwards(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, threshold: float = 1.6) -> torch.Tensor:
    """Checks whether the TCP is facing backwards (yaw error > ~90 degrees) compared to the commanded orientation."""

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
    return torch.abs(yaw_error) > threshold

def tcp_above_ground(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """Checks whether the TCP is above the ground (height > threshold).
        threshold means the maxmum height of the mowing head, the obstacle is blow this height, which mowing head can pass over
    """

    asset = env.scene[asset_cfg.name]

    # current position
    curr_pos_w = asset.data.body_link_state_w[:, asset_cfg.body_ids[0], :3]


    print("TCP height: ", curr_pos_w[:, 2])
    return curr_pos_w[:, 2] > threshold

def undesired_collision(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 100000.0) -> torch.Tensor:
    """
    Checks for undesired collisions based on contact sensor data.
    """

    contact_sensor = env.scene[sensor_cfg.name]

    forces = contact_sensor.data.net_forces_w # [N, 1, 3]

    
    impact_magnitude = torch.norm(forces, dim=-1)

    max_impact = torch.max(impact_magnitude, dim=1)[0] # if more than one contact sensor

    return max_impact > threshold

class tcp_hovering_above_terrain_termination(ManagerTermBase):
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
        return self.timer > time_threshold_sec
    
    def reset(self, env_ids=None):
        if env_ids is None:
            self.timer[:] = 0.0
            self.above_state[:] = False
        else:
            self.timer[env_ids] = 0.0
            self.above_state[env_ids] = False


class tcp_off_working_space_termination(ManagerTermBase):
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
        return self.timer > time_threshold_sec
    
    def reset(self, env_ids=None):
        if env_ids is None:
            self.timer[:] = 0.0
            self.above_state[:] = False
        else:
            self.timer[env_ids] = 0.0
            self.above_state[env_ids] = False