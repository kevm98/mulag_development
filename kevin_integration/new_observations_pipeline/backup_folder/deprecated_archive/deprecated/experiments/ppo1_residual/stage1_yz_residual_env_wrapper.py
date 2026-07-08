"""Stage-1 Y-Z residual PPO wrapper — base greedy + PPO residual over bias backend."""
from __future__ import annotations
from pathlib import Path
import numpy as np, torch, os, math
import gymnasium as gym

from isaaclab.utils.math import quat_apply_inverse
from rsl_rl.env import VecEnv
from tensordict import TensorDict

# ---- Fixed mapper constants ----
J_yz = torch.tensor([
    [-1.74,  29.81,  20.45,  20.29],
    [-1.51, -24.73, -30.88, -31.66],
], dtype=torch.float64)

LAMBDA = 1.0
MAX_CYL_DELTA = 0.005
SUCCESS_TOLERANCE = 0.10

# Residual-specific OBS_DIM: adds base_action [2] to standard obs
OBS_DIM = 8  # [target_error_yz(2), target_vel_yz(2), prev_processed_action(2), base_action(2)]


class Stage1YZResidualEnvWrapper(VecEnv):
    """Residual PPO over greedy base controller. Base is a P-controller on task-space error.
    PPO learns a small additive correction.
    """

    def __init__(self, cfg, horizon_steps=10, action_scale_m=0.05, device="cuda:0",
                 kp_scale=None, kd_scale=None, force_rate_limit=None, velocity_filter_beta=None,
                 trajectory_mode="sequence", target_hold_steps=50,
                 post_reset_settle_steps=5,
                 backend="explicit_effort_pid_bias",
                 bias_calibration_time_s=5.0,
                 debug_bias_shapes=False,
                 reward_mode="distance_only",
                 action_penalty_weight=0.005,
                 ee_jump_penalty_weight=0.05,
                 action_smoothing_beta=0.8,
                 action_rate_limit=0.1,
                 base_gain=1.0,
                 residual_scale=0.05):
        cfg.scene.num_envs = cfg.scene.num_envs
        cfg.enable_scripted_vehicle_motion = False
        cfg.enable_startup_summary = False
        cfg.log_target_delta_xdot_clipping = False
        cfg.allow_debug_manual_cylinder_targets = True
        cfg.lowlevel_actuator_backend = str(backend)
        cfg.explicit_pid_kp_scale = float(kp_scale) if kp_scale is not None else 0.1
        cfg.explicit_pid_kd_scale = float(kd_scale) if kd_scale is not None else 0.5
        cfg.explicit_pid_effort_scale = 1.0
        cfg.explicit_force_rate_limit_fraction = float(force_rate_limit) if force_rate_limit is not None else 0.01
        cfg.explicit_pd_velocity_filter_beta = float(velocity_filter_beta) if velocity_filter_beta is not None else 0.8
        cfg.bias_calibration_time_s = float(bias_calibration_time_s)
        cfg.bias_average_window_s = 0.5
        cfg.bias_calibration_ki_scale = 5.0
        cfg.bias_calibration_requires_static = False
        cfg.bias_runtime_ki_scale = 0.0
        cfg.bias_online_update_rate = 0.0
        cfg.debug_bias_shapes = bool(debug_bias_shapes)
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
        self._action_scale = float(action_scale_m)
        self._device = device
        self._cyl_ids = self._uw._cylinder_joint_ids
        self._tool_id = self._uw._tool_joint_id
        self._ee_id = self._uw._ee_body_id
        self._dt = float(self._uw.step_dt)

        B = self.num_envs = self._uw.num_envs
        self.device = device
        self.max_episode_length = 200
        self.num_actions = 2
        self.num_obs = OBS_DIM

        self._trajectory_mode = trajectory_mode
        self._target_hold_steps = int(target_hold_steps)
        self._post_reset_settle_steps = int(post_reset_settle_steps)
        self._target_y_range = 0.05
        self._target_z_range = 0.02

        # Jacobian pseudo-inverse for cylinder mapping
        JT = J_yz.T.to(torch.float64)
        I2 = torch.eye(2, dtype=torch.float64)
        self._J_pinv = (JT @ torch.linalg.inv(J_yz.to(torch.float64) @ JT + LAMBDA**2 * I2)).float().to(device)

        # Residual controller parameters
        self._base_gain = float(base_gain)
        self._residual_scale = float(residual_scale)
        self._reward_mode = str(reward_mode)
        self._action_penalty_weight = float(action_penalty_weight)
        self._ee_jump_penalty_weight = float(ee_jump_penalty_weight)
        self._action_smoothing_beta = float(action_smoothing_beta)
        self._action_rate_limit = float(action_rate_limit)

        # State
        self._target_yz = torch.zeros((B, 2), device=device)
        self._target_vel_yz = torch.zeros((B, 2), device=device)
        self._prev_action = torch.zeros((B, 2), device=device)  # raw PPO action
        self._processed_action = torch.zeros((B, 2), device=device)  # smoothed total action
        self._base_action = torch.zeros((B, 2), device=device)  # base controller output
        self._ep_length = torch.zeros(B, dtype=torch.long, device=device)
        self._prev_distance = torch.zeros(B, device=device)
        self._root_pose = None
        self._root_vel = torch.zeros((B, 6), device=device)
        self._tool_target = None
        self._ep_return = torch.zeros(B, device=device)
        self._base_root_pose = None  # stored after reset settle

        # Trajectory state
        self._traj_t = torch.zeros(B, device=device)
        self._traj_mode_per_env = [trajectory_mode] * B
        self._traj_params = [None] * B

        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, B)
        print(f"[ResidualYZ] H={self._H} act_scale={self._action_scale}m obs={OBS_DIM} act=2 "
              f"base_gain={self._base_gain} residual_scale={self._residual_scale} "
              f"mode={trajectory_mode} backend={backend}", flush=True)

    def _tcp_yz(self):
        ee_w = self._uw.robot.data.body_link_state_w[:, self._ee_id, :3]
        bp = self._uw.robot.data.root_pos_w
        bq = self._uw.robot.data.root_quat_w
        bq = bq / bq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return quat_apply_inverse(bq, ee_w - bp)[:, 1:3]

    def _pin_root(self):
        """Write current root pose back to sim to keep base stationary."""
        if self._base_root_pose is not None:
            self._uw.robot.write_root_pose_to_sim(self._base_root_pose)
            self._uw.robot.write_root_velocity_to_sim(
                torch.zeros((self.num_envs, 6), device=self._device))

    def _init_trajectory(self, ee, mode):
        p = {}
        p['mode'] = mode
        p['center'] = ee.clone()
        if mode == 'sequence':
            p['hold_left'] = self._target_hold_steps
        elif mode == 'line':
            dy = (torch.rand(1, device=self._device) * 0.08 - 0.04).item()
            dz = (torch.rand(1, device=self._device) * 0.03 - 0.015).item()
            p['start'] = ee.clone()
            p['end'] = ee + torch.tensor([dy, dz], device=self._device)
            p['duration'] = 100
            p['t'] = 0
        elif mode == 'circle':
            p['radius_y'] = 0.04
            p['radius_z'] = 0.015
            p['omega'] = 0.03
            p['t'] = 0
        elif mode == 'curve':
            p['amp_y'] = 0.04
            p['amp_z'] = 0.015
            p['freq'] = 0.02
            p['t'] = 0
        return p

    def _step_trajectory(self, env_idx):
        B = 1
        ee = self._tcp_yz()[env_idx:env_idx+1]
        mode = self._traj_mode_per_env[env_idx]
        params = self._traj_params[env_idx]

        if mode == 'sequence':
            if params['hold_left'] <= 0:
                yr = self._target_y_range; zr = self._target_z_range
                dy = (torch.rand(1, device=self._device) * 2 * yr - yr)
                dz = (torch.rand(1, device=self._device) * 2 * zr - zr)
                self._target_yz[env_idx] = (ee + torch.cat([dy, dz], dim=-1))[0]
                self._target_vel_yz[env_idx].zero_()
                params['hold_left'] = self._target_hold_steps
                self._prev_distance[env_idx] = torch.norm(self._target_yz[env_idx:env_idx+1] - ee, dim=-1)
            else:
                params['hold_left'] -= 1
                self._target_vel_yz[env_idx].zero_()

        elif mode == 'line':
            t = params['t'] / params['duration']
            t = min(max(t, 0.0), 1.0)
            self._target_yz[env_idx] = (params['start'] * (1 - t) + params['end'] * t)[0]
            self._target_vel_yz[env_idx] = ((params['end'] - params['start']) / max(params['duration'], 1))[0]
            params['t'] += 1
            if params['t'] >= params['duration']:
                params['t'] = 0
                params['start'], params['end'] = params['end'], params['start']

        elif mode == 'circle':
            t = params['t'] * params['omega']
            target = params['center'] + torch.tensor([params['radius_y'] * math.cos(t),
                                                       params['radius_z'] * math.sin(t)], device=self._device)
            vel = torch.tensor([-params['radius_y'] * params['omega'] * math.sin(t),
                                 params['radius_z'] * params['omega'] * math.cos(t)], device=self._device)
            self._target_yz[env_idx] = target
            self._target_vel_yz[env_idx] = vel
            params['t'] += 1

        elif mode == 'curve':
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

    def _compute_base_action(self):
        """Greedy P-controller: base_action = clamp(Kg * error / action_scale, -1, 1)."""
        err = self._target_yz - self._tcp_yz()
        return torch.clamp(self._base_gain * err / self._action_scale, -1.0, 1.0)

    def _compute_reward(self, total_action, residual_raw):
        ee = self._tcp_yz()
        d = torch.norm(ee - self._target_yz, dim=-1)
        p = self._prev_distance - d
        an_total = torch.norm(total_action, dim=-1)
        an_res = torch.norm(residual_raw, dim=-1)
        if self._reward_mode == "distance_only":
            r = (-1.0 * d
                 - self._action_penalty_weight * (an_total ** 2)
                 - 0.005 * (an_res ** 2))
        else:
            r = 10.0 * p - 1.0 * d - 0.001 * an_total - 0.005 * (an_res ** 2)
        self._prev_distance = d.detach()
        return r, d, (d < SUCCESS_TOLERANCE), p

    def reset(self):
        self._env.reset()
        B = self.num_envs
        self._root_pose = self._uw.robot.data.root_state_w[:, :7].clone()
        self._root_vel.zero_()
        cqh = self._uw.robot.data.joint_pos[:, self._cyl_ids].clone()
        self._tool_target = self._uw.robot.data.joint_pos[:, self._tool_id].unsqueeze(-1).clone()
        for _ in range(self._post_reset_settle_steps):
            self._uw.set_debug_manual_cylinder_targets(cqh, tool_position_target=self._tool_target)
            self._env.step(torch.zeros((B, 5), device=self._device))
        ee = self._tcp_yz()
        self._base_root_pose = self._uw.robot.data.root_state_w[:, :7].clone()
        self._prev_action.zero_()
        self._processed_action.zero_()
        self._base_action.zero_()
        self._ep_length.zero_()
        self._ep_return.zero_()
        self._traj_t.zero_()

        modes_list = ['sequence', 'line', 'circle', 'curve']
        for i in range(B):
            if self._trajectory_mode == 'mixed':
                mode = modes_list[i % len(modes)]
            else:
                mode = self._trajectory_mode
            self._traj_mode_per_env[i] = mode
            self._traj_params[i] = self._init_trajectory(ee[i:i+1], mode)

        for i in range(B):
            mode = self._traj_mode_per_env[i]
            if mode == 'sequence':
                yr = self._target_y_range; zr = self._target_z_range
                dy = (torch.rand(1, device=self._device) * 2 * yr - yr)
                dz = (torch.rand(1, device=self._device) * 2 * zr - zr)
                self._target_yz[i] = ee[i] + torch.cat([dy, dz], dim=-1)
                self._target_vel_yz[i].zero_()
            else:
                self._step_trajectory(i)

        self._prev_distance = torch.norm(self._tcp_yz() - self._target_yz, dim=-1)
        return self._get_obs(), {}

    def _get_obs(self):
        err = self._target_yz - self._tcp_yz()
        return torch.cat([err, self._target_vel_yz, self._processed_action, self._base_action], dim=-1)

    def _to_tensordict(self, obs):
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=obs.device)

    def get_observations(self):
        return self._to_tensordict(self._get_obs())

    def step(self, residual_raw):
        B = self.num_envs
        residual_raw = torch.nan_to_num(residual_raw, nan=0.0).clamp(-1.0, 1.0)

        # Compute base controller action
        self._base_action = self._compute_base_action().detach()

        # Total action = base + residual_scale * residual_raw
        raw_total = self._base_action + self._residual_scale * residual_raw

        # Action smoothing and rate limiting
        processed = torch.clamp(raw_total, -1.0, 1.0)
        if self._action_rate_limit > 0:
            delta = processed - self._processed_action
            delta = torch.clamp(delta, -self._action_rate_limit, self._action_rate_limit)
            processed = self._processed_action + delta
        if self._action_smoothing_beta > 0:
            beta = self._action_smoothing_beta
            processed = beta * self._processed_action + (1.0 - beta) * processed
        self._processed_action = processed.detach().clone()
        self._prev_action = residual_raw.detach().clone()

        # Map to cylinder targets through Jacobian
        dc = torch.clamp((self._J_pinv @ (processed * self._action_scale).T).T,
                         -MAX_CYL_DELTA, MAX_CYL_DELTA)
        cqh = self._uw.robot.data.joint_pos[:, self._cyl_ids].clone()
        lim = self._uw.robot.data.soft_joint_pos_limits[:, self._cyl_ids]
        cqt = torch.clamp(cqh + dc, lim[..., 0], lim[..., 1])

        ee0 = self._tcp_yz().detach().clone()
        for _ in range(self._H):
            self._uw.set_debug_manual_cylinder_targets(cqt, tool_position_target=self._tool_target)
            self._env.step(torch.zeros((B, 5), device=self._device))
            self._pin_root()
        ee1 = self._tcp_yz()
        jump = torch.norm(ee1 - ee0, dim=-1)

        reward, distance, success, progress = self._compute_reward(processed, residual_raw)
        reward = reward - self._ee_jump_penalty_weight * jump

        for i in range(B):
            self._step_trajectory(i)

        self._ep_length += 1
        term = (jump > 0.30)
        trunc = self._ep_length >= self.max_episode_length
        done = term | trunc
        self._ep_return += reward

        raw_anorm = torch.norm(residual_raw, dim=-1).mean()
        proc_anorm = torch.norm(processed, dim=-1).mean()
        base_anorm = torch.norm(self._base_action, dim=-1).mean()
        target_speed = torch.norm(self._target_vel_yz, dim=-1).mean()

        extras = {"log": {
            "distance": distance.mean(),
            "success_rate": success.float().mean(),
            "ee_jump": jump.mean(),
            "raw_action_norm": raw_anorm,
            "processed_action_norm": proc_anorm,
            "base_action_norm": base_anorm,
            "reward_mean": reward.mean(),
            "progress": progress.mean(),
            "target_speed": target_speed,
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
            self._base_root_pose[dids] = self._uw.robot.data.root_state_w[dids, :7].clone()
            self._prev_action[dids] = 0.0
            self._processed_action[dids] = 0.0
            self._base_action[dids] = 0.0
            ee_reset = self._tcp_yz()[dids]
            modes_list = ['sequence', 'line', 'circle', 'curve']
            for j, idx in enumerate(dids.tolist()):
                i = int(idx)
                if self._trajectory_mode == 'mixed':
                    mode = modes_list[i % len(modes_list)]
                else:
                    mode = self._trajectory_mode
                self._traj_mode_per_env[i] = mode
                self._traj_params[i] = self._init_trajectory(ee_reset[j:j+1], mode)
                if mode == 'sequence':
                    yr = self._target_y_range; zr = self._target_z_range
                    dy = (torch.rand(1, device=self._device) * 2 * yr - yr)
                    dz = (torch.rand(1, device=self._device) * 2 * zr - zr)
                    self._target_yz[i] = ee_reset[j] + torch.cat([dy, dz], dim=-1)
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
