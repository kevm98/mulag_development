"""Stage-1 Y-Z trajectory tracking wrapper with stationary base, moving targets."""
from __future__ import annotations
from pathlib import Path
import numpy as np, torch, os, math
import gymnasium as gym

from isaaclab.utils.math import quat_apply_inverse
from rsl_rl.env import VecEnv
from tensordict import TensorDict

# ---- Fixed mapper constants (same as Stage 0) ----
# Original pinned J_yz (3/4 action sign passing, +Z limited by gravity)
J_yz = torch.tensor([
    [-1.74,  29.81,  20.45,  20.29],
    [-1.51, -24.73, -30.88, -31.66],
], dtype=torch.float64)

LAMBDA = 1.0
MAX_CYL_DELTA = 0.005
SUCCESS_TOLERANCE = 0.10
OBS_DIM = 6  # [target_error_yz(2), target_vel_yz(2), prev_action(2)]


class Stage1YZTrackingEnvWrapper(VecEnv):
    """Y-Z trajectory tracking with moving targets, stationary vehicle base."""

    def __init__(self, cfg, horizon_steps=10, action_scale_m=0.05, device="cuda:0",
                 kp_scale=None, kd_scale=None, force_rate_limit=None, velocity_filter_beta=None,
                 trajectory_mode="mixed", target_hold_steps=30,
                 post_reset_settle_steps=30,
                 backend="explicit_effort_pid_bias",
                 bias_calibration_time_s=2.0,
                 debug_bias_shapes=False,
                 reward_mode="default",
                 action_penalty_weight=0.005,
                 ee_jump_penalty_weight=0.05,
                 action_smoothing_beta=0.0,
                 action_rate_limit=-1.0):
        cfg.scene.num_envs = cfg.scene.num_envs
        cfg.enable_scripted_vehicle_motion = False
        cfg.enable_startup_summary = False
        cfg.log_target_delta_xdot_clipping = False
        cfg.allow_debug_manual_cylinder_targets = True
        cfg.lowlevel_actuator_backend = str(backend)
        # PID / bias backend config
        cfg.explicit_pid_kp_scale = float(kp_scale) if kp_scale is not None else 0.1
        cfg.explicit_pid_kd_scale = float(kd_scale) if kd_scale is not None else 0.5
        cfg.explicit_pid_effort_scale = 1.0
        cfg.explicit_force_rate_limit_fraction = float(force_rate_limit) if force_rate_limit is not None else 0.01
        cfg.explicit_pd_velocity_filter_beta = float(velocity_filter_beta) if velocity_filter_beta is not None else 0.8
        cfg.explicit_force_rate_limit_fraction = float(force_rate_limit) if force_rate_limit is not None else 0.01
        # Bias calibration config
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

        JT = J_yz.T.to(torch.float64)
        I2 = torch.eye(2, dtype=torch.float64)
        self._J_pinv = (JT @ torch.linalg.inv(J_yz.to(torch.float64) @ JT + LAMBDA**2 * I2)).float().to(device)

        # Action calibration: pre-multiply policy actions to decouple Y/Z response
        # Measured R at N=5 (no-pin, kp=0.1, kd=0.5): R = [[0.0236, 0.0098], [0.0078, 0.0069]]
        # Using desired_scale=0.005m per unit action to avoid saturation
        self._action_calibration = torch.tensor([
            [ 0.4008, -0.5709],
            [-0.4580,  1.3808],
        ], device=device).float()

        # Trajectory state
        self._target_yz = torch.zeros((B, 2), device=device)
        self._target_vel_yz = torch.zeros((B, 2), device=device)
        self._prev_action = torch.zeros((B, 2), device=device)
        self._ep_length = torch.zeros(B, dtype=torch.long, device=device)
        self._prev_distance = torch.zeros(B, device=device)
        self._root_pose = None
        self._root_vel = torch.zeros((B, 6), device=device)
        self._base_root_pose = None  # stored after reset settle
        self._tool_target = None
        self._ep_return = torch.zeros(B, device=device)
        self._reward_mode = str(reward_mode)
        self._action_penalty_weight = float(action_penalty_weight)
        self._ee_jump_penalty_weight = float(ee_jump_penalty_weight)
        self._action_smoothing_beta = float(action_smoothing_beta)
        self._action_rate_limit = float(action_rate_limit)
        self._processed_action = torch.zeros((B, 2), device=device)

        # Trajectory generators state (per env)
        self._traj_t = torch.zeros(B, device=device)
        self._traj_mode_per_env = [trajectory_mode] * B  # string per env
        self._traj_params = [None] * B  # per-env trajectory params

        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, B)
        print(f"[Stage1YZ] H={self._H} act_scale={self._action_scale}m obs={OBS_DIM} act=2 mode={trajectory_mode} backend={backend}", flush=True)

    _debug_enabled = False
    _debug_steps = 3

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

    def _sample_local_target(self):
        """Sample a local static target around current EE."""
        B = self.num_envs
        ee = self._tcp_yz()
        dy = (torch.rand(B, device=self._device) * 0.10 - 0.05)
        dz = (torch.rand(B, device=self._device) * 0.04 - 0.02)
        self._target_yz = ee + torch.stack([dy, dz], dim=-1)
        self._target_vel_yz.zero_()

    def _init_trajectory(self, ee, mode):
        """Initialize a trajectory generator for a single env. Returns params dict."""
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
        """Advance trajectory for one env, update target_yz and target_vel_yz.
        Returns True if target changed significantly (resample / large jump)."""
        B = 1
        ee = self._tcp_yz()[env_idx:env_idx+1]
        prev_target = self._target_yz[env_idx:env_idx+1].clone()
        mode = self._traj_mode_per_env[env_idx]
        params = self._traj_params[env_idx]
        target_changed = False

        if mode == 'sequence':
            if params['hold_left'] <= 0:
                yr = self._target_y_range; zr = self._target_z_range
                dy = (torch.rand(1, device=self._device) * 2 * yr - yr)
                dz = (torch.rand(1, device=self._device) * 2 * zr - zr)
                self._target_yz[env_idx] = (ee + torch.cat([dy, dz], dim=-1))[0]
                self._target_vel_yz[env_idx].zero_()
                params['hold_left'] = self._target_hold_steps
                target_changed = True
                # Reset prev_distance for this env to new target distance
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

        # Clamp target velocity
        max_vel = 0.003
        if target_changed:
            self._prev_distance[env_idx] = torch.norm(self._target_yz[env_idx:env_idx+1] - ee, dim=-1)
        vnorm = torch.norm(self._target_vel_yz[env_idx:env_idx+1], dim=-1)
        if vnorm > max_vel:
            self._target_vel_yz[env_idx] *= max_vel / vnorm

    def _compute_reward(self, action):
        ee = self._tcp_yz()
        d = torch.norm(ee - self._target_yz, dim=-1)
        p = self._prev_distance - d  # logged but may not be used in reward
        an = torch.norm(action, dim=-1)
        if self._reward_mode == "distance_only":
            r = -1.0 * d - self._action_penalty_weight * (an ** 2)
        else:
            r = 10.0 * p - 1.0 * d - 0.001 * an
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
        self._ep_length.zero_()
        self._ep_return.zero_()
        self._traj_t.zero_()

        # Initialize trajectory per env
        modes_list = ['sequence', 'line', 'circle', 'curve']
        for i in range(B):
            if self._trajectory_mode == 'mixed':
                mode = modes_list[i % len(modes)]
            else:
                mode = self._trajectory_mode
            self._traj_mode_per_env[i] = mode
            self._traj_params[i] = self._init_trajectory(ee[i:i+1], mode)

        # Set initial targets: for sequence, sample local target; for others, step trajectory
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
        if self._debug_enabled:
            ee0 = self._tcp_yz()[0]; t0 = self._target_yz[0]
            print(f"[S1_DEBUG] RESET: ee=({ee0[0].item():.4f},{ee0[1].item():.4f})  "
                  f"target=({t0[0].item():.4f},{t0[1].item():.4f})  "
                  f"err_norm={torch.norm(t0-ee0).item():.4f}  mode={self._traj_mode_per_env[0]}",
                  flush=True)
        return self._get_obs(), {}

    def _get_obs(self):
        err = self._target_yz - self._tcp_yz()
        return torch.cat([err, self._target_vel_yz, self._processed_action], dim=-1)

    def _to_tensordict(self, obs):
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=obs.device)

    def get_observations(self):
        return self._to_tensordict(self._get_obs())

    def step(self, action):
        B = self.num_envs
        raw_action = torch.nan_to_num(action, nan=0.0).clamp(-1.0, 1.0)

        # --- Action smoothing and rate limiting ---
        processed = raw_action.clone()
        if self._action_rate_limit > 0:
            delta = raw_action - self._processed_action
            delta = torch.clamp(delta, -self._action_rate_limit, self._action_rate_limit)
            processed = self._processed_action + delta
        if self._action_smoothing_beta > 0:
            beta = self._action_smoothing_beta
            processed = beta * self._processed_action + (1.0 - beta) * processed
        self._processed_action = processed.detach().clone()
        self._prev_action = raw_action.detach().clone()

        # --- Debug: first few steps after reset ---
        if self._debug_enabled and self._ep_length[0].item() < self._debug_steps:
            ee = self._tcp_yz()[0]
            target = self._target_yz[0]
            tvel = self._target_vel_yz[0]
            pa = self._processed_action[0]
            err = target - ee
            mode = self._traj_mode_per_env[0]
            phase = self._traj_params[0].get('t', self._traj_params[0].get('hold_left', '?'))
            print(f"[S1_DEBUG] step={self._ep_length[0].item()} mode={mode} phase={phase}  "
                  f"ee=({ee[0].item():.4f},{ee[1].item():.4f})  "
                  f"target=({target[0].item():.4f},{target[1].item():.4f})  "
                  f"err=({err[0].item():.4f},{err[1].item():.4f})  "
                  f"err_norm={torch.norm(err).item():.4f}  "
                  f"tvel=({tvel[0].item():.5f},{tvel[1].item():.5f})  "
                  f"raw_a=({raw_action[0,0].item():.3f},{raw_action[0,1].item():.3f})  "
                  f"proc_a=({pa[0].item():.3f},{pa[1].item():.3f})",
                  flush=True)
        dc = torch.clamp((self._J_pinv @ (processed * self._action_scale).T).T, -MAX_CYL_DELTA, MAX_CYL_DELTA)
        if self._debug_enabled and self._ep_length[0].item() < self._debug_steps:
            print(f"[S1_DEBUG] mapped_cyl_delta=({dc[0,0].item():.6f},{dc[0,1].item():.6f},{dc[0,2].item():.6f},{dc[0,3].item():.6f})",
                  flush=True)
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

        # Compute reward BEFORE advancing trajectory (against current target)
        reward, distance, success, progress = self._compute_reward(processed)
        # Add jump penalty using configured weight
        reward = reward - self._ee_jump_penalty_weight * jump
        if self._debug_enabled and self._ep_length[0].item() < self._debug_steps:
            pd = self._prev_distance[0].item()
            print(f"[S1_DEBUG] prev_dist={pd:.4f} dist={distance[0].item():.4f} "
                  f"progress={progress[0].item():.4f} success={success[0].item()} "
                  f"jump={jump[0].item():.4f} reward={reward[0].item():.4f}",
                  flush=True)

        # Advance trajectory AFTER reward computation
        for i in range(B):
            self._step_trajectory(i)

        self._ep_length += 1
        term = (jump > 0.30)
        trunc = self._ep_length >= self.max_episode_length
        done = term | trunc
        self._ep_return += reward
        target_speed = torch.norm(self._target_vel_yz, dim=-1).mean()
        raw_anorm = torch.norm(raw_action, dim=-1).mean()
        proc_anorm = torch.norm(processed, dim=-1).mean()
        delta_anorm = torch.norm(raw_action - processed, dim=-1).mean()
        extras = {"log": {"distance": distance.mean(), "success_rate": success.float().mean(),
                          "ee_jump": jump.mean(),
                          "raw_action_norm": raw_anorm,
                          "processed_action_norm": proc_anorm,
                          "action_delta_norm": delta_anorm,
                          "reward_mean": reward.mean(), "progress": progress.mean(),
                          "target_speed": target_speed,
                          "episode_return": self._ep_return.mean()}}
        dids = done.nonzero(as_tuple=False).squeeze(-1)
        if dids.numel() > 0:
            if dids.ndim == 0:
                dids = dids.unsqueeze(0)
            self._uw._reset_idx(dids.tolist())
            # Settle arm after reset before continuing
            cqh_s = self._uw.robot.data.joint_pos[:, self._cyl_ids].clone()
            tool_s = self._uw.robot.data.joint_pos[:, self._tool_id].unsqueeze(-1).clone()
            for _ in range(self._post_reset_settle_steps):
                self._uw.set_debug_manual_cylinder_targets(cqh_s, tool_position_target=tool_s)
                self._env.step(torch.zeros((B, 5), device=self._device))
            self._base_root_pose[dids] = self._uw.robot.data.root_state_w[dids, :7].clone()
            self._prev_action[dids] = 0.0
            self._processed_action[dids] = 0.0
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
                # Set initial target per-env
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
