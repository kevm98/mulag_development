# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

class random_driving_speed(ManagerTermBase):
    def __init__(self, cfg: SceneEntityCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.wheel_radius = 0.575 # radius of unimog wheel 57.5 cm
        self.vel_range = cfg.params["velocity_range"]
        self.asset = env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(cfg.params["joint_names"])

        if len(self.joint_ids) == 0:
            raise ValueError(f"Could not find joints for asset: {cfg.asset_name}")
        
        self.commanded_velocities = torch.zeros(
            (env.num_envs, len(self.joint_ids)), 
            device=env.device
        )

    def __call__(self, 
                 env: ManagerBasedRLEnv, 
                 env_ids: torch.Tensor | None, 
                 joint_names: list[str] = None, 
                 velocity_range: tuple[float, float] = None):


        if env_ids is None:
            env_ids = env.scene.env_ids


        current_episode_lengths = env.episode_length_buf[env_ids]
        reset_mask = current_episode_lengths <= 10 # only resample speed of terminated envs
        ids_to_resample = env_ids[reset_mask]

        if len(ids_to_resample) > 0:
            lower, upper = self.vel_range
            
            rand_val = torch.rand((len(ids_to_resample), 1), device=env.device) # (N, 1) 
            sampled_velocity_kmh = lower + (upper - lower) * rand_val

            # km/h -> m/s -> rad/s
            # v_ms = v_kmh / 3.6
            # omega = v_ms / radius
            sampled_velocity_rads = (sampled_velocity_kmh / 3.6) / self.wheel_radius
            
            velocity_targets_new = sampled_velocity_rads.expand(-1, len(self.joint_ids)) # (N, num_joints)
            
            self.commanded_velocities[ids_to_resample] = velocity_targets_new
            #print(f"Resampled velocities for {len(ids_to_resample)} envs.")

        targets_to_apply = self.commanded_velocities[env_ids]

        self.asset.write_joint_velocity_to_sim(
            targets_to_apply, 
            joint_ids=self.joint_ids, 
            env_ids=env_ids
        )