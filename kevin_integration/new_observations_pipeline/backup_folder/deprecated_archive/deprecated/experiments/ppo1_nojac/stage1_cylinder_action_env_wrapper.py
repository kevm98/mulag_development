"""Stage-1 no-Jacobian PPO — policy outputs 4D cylinder actions directly.
No J_yz, no DLS inverse. Policy learns actuator coupling from experience.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, torch, math, json, os
import gymnasium as gym

from isaaclab.utils.math import quat_apply_inverse
from rsl_rl.env import VecEnv
from tensordict import TensorDict

OBS_DIM = 18
ACT_DIM = 4
SUCCESS_TOLERANCE = 0.10


class Stage1CylinderActionEnvWrapper(VecEnv):
    """No-Jacobian cylinder-direct PPO env. Action = [4] cylinder delta in [-1,1]."""

    def __init__(self, cfg, horizon_steps=10, action_scale_m=None, device="cuda:0",
                 kp_scale=None, kd_scale=None,
                 trajectory_mode="sequence_easy", target_hold_steps=50,
                 post_reset_settle_steps=5,
                 backend="explicit_effort_pid_bias",
                 bias_calibration_time_s=5.0,
                 max_cylinder_delta_m=0.0025,
                 action_smoothing_beta=0.8,
                 action_rate_limit=0.1,
                 reward_mode="distance_only",
                 workspace_yz=None,  # dict from workspace JSON
                 ee_vel_scale=0.05,
                 cyl_vel_scale=0.05):
        cfg.scene.num_envs = cfg.scene.num_envs
        cfg.enable_scripted_vehicle_motion = False
        cfg.enable_startup_summary = False
        cfg.allow_debug_manual_cylinder_targets = True
        cfg.lowlevel_actuator_backend = str(backend)
        cfg.explicit_pid_kp_scale = float(kp_scale) if kp_scale is not None else 0.1
        cfg.explicit_pid_kd_scale = float(kd_scale) if kd_scale is not None else 0.5
        cfg.explicit_force_rate_limit_fraction = 0.01
        cfg.explicit_pd_velocity_filter_beta = 0.8
        cfg.bias_calibration_time_s = float(bias_calibration_time_s)
        cfg.bias_average_window_s = 0.5
        cfg.bias_calibration_ki_scale = 5.0
        cfg.bias_calibration_requires_static = False
        cfg.bias_runtime_ki_scale = 0.0
        cfg.bias_online_update_rate = 0.0
        cfg.debug_bias_shapes = False
        if device:
            cfg.device = device
            cfg.sim.device = device

        from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
        from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

        id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
        cfg.id_checkpoint_override_path = str(id_path)
        apply_kevin_sim_memory_optimizations(cfg.sim)

        self.cfg = cfg
        self._env = MulagUpperLevelEnv(cfg=cfg)
        self._uw = self._env.unwrapped if hasattr(self._env, 'unwrapped') else self._env
        self._H = int(horizon_steps)
        self._device = device
        self._cyl_ids = self._uw._cylinder_joint_ids
        self._tool_id = self._uw._tool_joint_id
        self._ee_id = self._uw._ee_body_id

        B = self.num_envs = self._uw.num_envs
        self.device = device
        self.max_episode_length = 200
        self.num_actions = ACT_DIM
        self.num_obs = OBS_DIM

        self._trajectory_mode = trajectory_mode
        self._target_hold_steps = int(target_hold_steps)
        self._post_reset_settle_steps = int(post_reset_settle_steps)
        self._max_cyl_delta = float(max_cylinder_delta_m)
        self._action_smoothing_beta = float(action_smoothing_beta)
        self._action_rate_limit = float(action_rate_limit)
        self._reward_mode = str(reward_mode)
        self._ee_vel_scale = float(ee_vel_scale)
        self._cyl_vel_scale = float(cyl_vel_scale)

        # Workspace
        self._ws = workspace_yz or {}
        self._target_y_range = self._ws.get("recommended_target_y_range", 0.03)
        self._target_z_range = self._ws.get("recommended_target_z_range", 0.01)
        self._ws_center = self._ws.get("center_yz", [0.0, 0.0])

        # Cylinder joint limits for normalization
        self._cyl_lo = self._uw.robot.data.soft_joint_pos_limits[0, self._cyl_ids].clone()
        self._cyl_hi = self._uw.robot.data.soft_joint_pos_limits[0, self._cyl_ids + self._uw.robot.data.soft_joint_pos_limits.shape[1] // 2].clone() if False else \
                       self._uw.robot.data.soft_joint_pos_limits[0, self._cyl_ids + 4].clone()  # hack: upper limits are in second half
        # Actually just use the tensor directly
        # Cylinder joint limits: soft_joint_pos_limits has shape [B, N, 2]
        limits = self._uw.robot.data.soft_joint_pos_limits[0, :, :]  # [N, 2]
        self._cyl_lo = limits[self._cyl_ids, 0].clone().to(device)
        self._cyl_hi = limits[self._cyl_ids, 1].clone().to(device)

        # State
        self._target_yz = torch.zeros((B, 2), device=device)
        self._target_vel_yz = torch.zeros((B, 2), device=device)
        self._prev_action = torch.zeros((B, ACT_DIM), device=device)
        self._processed_action = torch.zeros((B, ACT_DIM), device=device)
        self._ep_length = torch.zeros(B, dtype=torch.long, device=device)
        self._prev_distance = torch.zeros(B, device=device)
        self._prev_ee_yz = torch.zeros((B, 2), device=device)
        self._tool_target = None
        self._ep_return = torch.zeros(B, device=device)

        # Trajectory state
        self._traj_params = [None] * B
        self._traj_mode_per_env = [trajectory_mode] * B
        self._traj_t = torch.zeros(B, device=device)

        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, B)
        print(f"[CylAct] obs={OBS_DIM} act={ACT_DIM} max_delta={self._max_cyl_delta} "
              f"mode={trajectory_mode} backend={backend}", flush=True)

    def _tcp_yz(self):
        ee_w = self._uw.robot.data.body_link_state_w[:, self._ee_id, :3]
        bp = self._uw.robot.data.root_pos_w
        bq = self._uw.robot.data.root_quat_w
        bq = bq / bq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return quat_apply_inverse(bq, ee_w - bp)[:, 1:3]

    def _tcp_vel_yz(self):
        return self._prev_ee_yz  # placeholder until implemented properly

    def _normalize_cyl_pos(self, pos):
        """Normalize cylinder positions to [-1, 1]."""
        mid = (self._cyl_lo + self._cyl_hi) / 2
        half = (self._cyl_hi - self._cyl_lo) / 2
        half = torch.clamp(half, min=1e-6)
        return 2.0 * (pos - mid) / half

    def _get_obs(self):
        ee = self._tcp_yz()
        err = self._target_yz - ee
        ws_scale = max(self._target_y_range, self._target_z_range, 0.02)
        err_norm = err / ws_scale
        tvel_norm = self._target_vel_yz / 0.003  # max target speed

        # EE velocity (finite difference)
        ee_vel = ee - self._prev_ee_yz
        ee_vel = torch.clamp(ee_vel / self._ee_vel_scale, -1.0, 1.0)
        self._prev_ee_yz = ee.detach().clone()

        # Cylinder state
        cyl_pos = self._uw.robot.data.joint_pos[:, self._cyl_ids]
        cyl_vel = self._uw.robot.data.joint_vel[:, self._cyl_ids]
        cyl_pos_norm = torch.clamp(self._normalize_cyl_pos(cyl_pos), -1.0, 1.0)
        cyl_vel_norm = torch.clamp(cyl_vel / self._cyl_vel_scale, -1.0, 1.0)

        return torch.cat([
            err_norm,          # 2
            tvel_norm,         # 2
            ee_vel,            # 2
            cyl_pos_norm,      # 4
            cyl_vel_norm,      # 4
            self._processed_action,  # 4
        ], dim=-1)  # total 18

    def _to_tensordict(self, obs):
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=obs.device)

    def get_observations(self):
        return self._to_tensordict(self._get_obs())

    def _init_trajectory(self, ee, mode):
        p = {'mode': mode}
        if 'sequence' in mode:
            p['hold_left'] = self._target_hold_steps
        elif mode == 'line_max':
            eps = self._ws.get("line_endpoints", [[-0.02, 0.0], [0.02, 0.0]])
            p['start'] = torch.tensor(eps[0], device=self._device)
            p['end'] = torch.tensor(eps[1], device=self._device)
            p['duration'] = 150
            p['t'] = 0
        elif mode == 'ellipse_max':
            p['radius_y'] = self._ws.get("ellipse_radius_y", 0.02)
            p['radius_z'] = self._ws.get("ellipse_radius_z", 0.008)
            p['omega'] = 0.02
            p['t'] = 0
            p['center'] = torch.tensor(self._ws_center, device=self._device)
        elif 'curve' in mode:
            p['amp_y'] = self._ws.get("curve_amplitude_y", 0.02)
            p['amp_z'] = self._ws.get("curve_amplitude_z", 0.008)
            p['freq'] = 0.015
            p['t'] = 0
            p['center'] = torch.tensor(self._ws_center, device=self._device)
        else:
            p['hold_left'] = self._target_hold_steps
        return p

    def _step_trajectory(self, env_idx):
        mode = self._traj_mode_per_env[env_idx]
        params = self._traj_params[env_idx]

        if 'sequence' in mode:
            if params['hold_left'] <= 0:
                yr = self._target_y_range; zr = self._target_z_range
                dy = (torch.rand(1, device=self._device) * 2 * yr - yr)
                dz = (torch.rand(1, device=self._device) * 2 * zr - zr)
                self._target_yz[env_idx] = (torch.tensor(self._ws_center, device=self._device) +
                                             torch.cat([dy, dz], dim=-1))[0]
                self._target_vel_yz[env_idx].zero_()
                params['hold_left'] = self._target_hold_steps
            else:
                params['hold_left'] -= 1
                self._target_vel_yz[env_idx].zero_()

        elif mode == 'line_max':
            t = params['t'] / params['duration']
            t = min(max(t, 0.0), 1.0)
            self._target_yz[env_idx] = (params['start'] * (1 - t) + params['end'] * t)[0]
            self._target_vel_yz[env_idx] = ((params['end'] - params['start']) / max(params['duration'], 1))[0]
            params['t'] += 1
            if params['t'] >= params['duration']:
                params['t'] = 0
                params['start'], params['end'] = params['end'], params['start']

        elif mode == 'ellipse_max':
            t = params['t'] * params['omega']
            target = params['center'] + torch.tensor([params['radius_y'] * math.cos(t),
                                                       params['radius_z'] * math.sin(t)], device=self._device)
            vel = torch.tensor([-params['radius_y'] * params['omega'] * math.sin(t),
                                 params['radius_z'] * params['omega'] * math.cos(t)], device=self._device)
            self._target_yz[env_idx] = target
            self._target_vel_yz[env_idx] = vel
            params['t'] += 1

        elif 'curve' in mode:
            t = params['t'] * params['freq']
            target = params['center'] + torch.tensor([params['amp_y'] * math.sin(t),
                                                       params['amp_z'] * math.cos(2 * t)], device=self._device)
            vel = torch.tensor([params['amp_y'] * params['freq'] * math.cos(t),
                                -2 * params['amp_z'] * params['freq'] * math.sin(2 * t)], device=self._device)
            self._target_yz[env_idx] = target
            self._target_vel_yz[env_idx] = vel
            params['t'] += 1

        vnorm = torch.norm(self._target_vel_yz[env_idx:env_idx+1], dim=-1)
        max_vel = 0.003
        if vnorm > max_vel:
            self._target_vel_yz[env_idx] *= max_vel / vnorm

    def _compute_reward(self, action, prev_action):
        ee = self._tcp_yz()
        d = torch.norm(ee - self._target_yz, dim=-1)
        p = self._prev_distance - d
        an = torch.norm(action, dim=-1)
        da = torch.norm(action - prev_action, dim=-1)

        # Limit penalty
        cyl_pos = self._uw.robot.data.joint_pos[:, self._cyl_ids]
        lo_margin = (cyl_pos - self._cyl_lo.unsqueeze(0)) / (self._cyl_hi - self._cyl_lo).unsqueeze(0).clamp(min=1e-6)
        hi_margin = (self._cyl_hi.unsqueeze(0) - cyl_pos) / (self._cyl_hi - self._cyl_lo).unsqueeze(0).clamp(min=1e-6)
        near_limit = ((lo_margin < 0.02) | (hi_margin < 0.02)).any(dim=-1).float()

        # Success bonus
        bonus = torch.zeros_like(d)
        bonus = torch.where(d < 0.03, bonus + 2.0, bonus)
        bonus = torch.where((d >= 0.03) & (d < 0.05), bonus + 1.0, bonus)

        if self._reward_mode == "distance_only":
            r = (-1.0 * d + bonus
                 - 0.002 * (an ** 2)
                 - 0.005 * (da ** 2)
                 - 0.1 * near_limit)
        else:
            r = (10.0 * p - 1.0 * d + bonus
                 - 0.002 * (an ** 2)
                 - 0.005 * (da ** 2)
                 - 0.1 * near_limit)

        self._prev_distance = d.detach()
        return r, d, (d < SUCCESS_TOLERANCE), p

    def reset(self):
        self._env.reset()
        B = self.num_envs
        cqh = self._uw.robot.data.joint_pos[:, self._cyl_ids].clone()
        self._tool_target = self._uw.robot.data.joint_pos[:, self._tool_id].unsqueeze(-1).clone()
        for _ in range(self._post_reset_settle_steps):
            self._uw.set_debug_manual_cylinder_targets(cqh, tool_position_target=self._tool_target)
            self._env.step(torch.zeros((B, 5), device=self._device))
        ee = self._tcp_yz()
        self._prev_action.zero_()
        self._processed_action.zero_()
        self._ep_length.zero_()
        self._ep_return.zero_()
        self._prev_ee_yz = ee.detach().clone()

        for i in range(B):
            mode = self._trajectory_mode
            self._traj_mode_per_env[i] = mode
            self._traj_params[i] = self._init_trajectory(ee[i:i+1], mode)
            if 'sequence' in mode:
                yr = self._target_y_range; zr = self._target_z_range
                dy = (torch.rand(1, device=self._device) * 2 * yr - yr)
                dz = (torch.rand(1, device=self._device) * 2 * zr - zr)
                self._target_yz[i] = (torch.tensor(self._ws_center, device=self._device) +
                                       torch.cat([dy, dz], dim=-1))
                self._target_vel_yz[i].zero_()
            else:
                self._step_trajectory(i)

        self._prev_distance = torch.norm(self._tcp_yz() - self._target_yz, dim=-1)
        return self._get_obs(), {}

    def step(self, action):
        B = self.num_envs

        # Process action
        raw_action = torch.nan_to_num(action, nan=0.0).clamp(-1.0, 1.0)
        if raw_action.shape[-1] != ACT_DIM:
            raise ValueError(f"Expected action shape [*,{ACT_DIM}], got {raw_action.shape}")

        # Rate limit
        processed = raw_action.clone()
        if self._action_rate_limit > 0:
            delta = processed - self._processed_action
            delta = torch.clamp(delta, -self._action_rate_limit, self._action_rate_limit)
            processed = self._processed_action + delta
        # Smooth
        if self._action_smoothing_beta > 0:
            processed = (self._action_smoothing_beta * self._processed_action +
                         (1.0 - self._action_smoothing_beta) * processed)

        self._processed_action = processed.detach().clone()
        self._prev_action = raw_action.detach().clone()

        # Scale to cylinder deltas
        dc = processed * self._max_cyl_delta

        # Apply to cylinder positions
        cqh = self._uw.robot.data.joint_pos[:, self._cyl_ids].clone()
        lim = self._uw.robot.data.soft_joint_pos_limits[:, self._cyl_ids]
        cqt = torch.clamp(cqh + dc, lim[..., 0], lim[..., 1])

        # Step horizon
        ee0 = self._tcp_yz().detach().clone()
        for _ in range(self._H):
            self._uw.set_debug_manual_cylinder_targets(cqt, tool_position_target=self._tool_target)
            self._env.step(torch.zeros((B, 5), device=self._device))
        ee1 = self._tcp_yz()
        jump = torch.norm(ee1 - ee0, dim=-1)

        reward, distance, success, progress = self._compute_reward(processed, self._prev_action)
        reward = reward - 0.05 * jump

        for i in range(B):
            self._step_trajectory(i)

        self._ep_length += 1
        term = (jump > 0.30)
        trunc = self._ep_length >= self.max_episode_length
        done = term | trunc
        self._ep_return += reward

        an = torch.norm(processed, dim=-1).mean()
        extras = {"log": {
            "distance": distance.mean(),
            "success_rate": success.float().mean(),
            "ee_jump": jump.mean(),
            "action_norm": an,
            "reward_mean": reward.mean(),
            "progress": progress.mean(),
            "episode_return": self._ep_return.mean(),
        }}

        dids = done.nonzero(as_tuple=False).squeeze(-1)
        if dids.numel() > 0:
            if dids.ndim == 0:
                dids = dids.unsqueeze(0)
            self._uw._reset_idx(dids.tolist())
            cqh_s = self._uw.robot.data.joint_pos[:, self._cyl_ids].clone()
            tool_s = self._uw.robot.data.joint_pos[:, self._tool_id].unsqueeze(-1).clone()
            for _ in range(self._post_reset_settle_steps):
                self._uw.set_debug_manual_cylinder_targets(cqh_s, tool_position_target=tool_s)
                self._env.step(torch.zeros((B, 5), device=self._device))
            self._prev_action[dids] = 0.0
            self._processed_action[dids] = 0.0
            ee_reset = self._tcp_yz()[dids]
            for j, idx in enumerate(dids.tolist()):
                i = int(idx)
                mode = self._trajectory_mode
                self._traj_mode_per_env[i] = mode
                self._traj_params[i] = self._init_trajectory(ee_reset[j:j+1], mode)
                if 'sequence' in mode:
                    yr = self._target_y_range; zr = self._target_z_range
                    dy = (torch.rand(1, device=self._device) * 2 * yr - yr)
                    dz = (torch.rand(1, device=self._device) * 2 * zr - zr)
                    self._target_yz[i] = (torch.tensor(self._ws_center, device=self._device) +
                                           torch.cat([dy, dz], dim=-1))
                    self._target_vel_yz[i].zero_()
                else:
                    self._step_trajectory(i)
            self._prev_distance[dids] = torch.norm(self._tcp_yz()[dids] - self._target_yz[dids], dim=-1)
            self._ep_length[dids] = 0
            self._ep_return[dids] = 0.0

        return self._to_tensordict(self._get_obs()), reward, (term | trunc).to(torch.long), extras

    def seed(self, seed=-1):
        return self._env.seed(seed)

    @property
    def episode_length_buf(self):
        return self._ep_length

    @episode_length_buf.setter
    def episode_length_buf(self, v):
        self._ep_length = v

    def close(self):
        self._env.close()
