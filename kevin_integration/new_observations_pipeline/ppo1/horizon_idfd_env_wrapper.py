#!/usr/bin/env python3
"""Horizon-rate PPO environment wrapper with ID/FD closed-loop rollout.

Each env.step(action) executes H physics steps:
    action[0:4] in [-1,1] → delta_x_des_H = action * action_scale_m
    → closed-loop ID/FD rollout → delta_fd_H
    → delta_cmd_H = fd_gain[c] * delta_fd_H
    → smooth PhysX ramp over H steps
    → horizon-level observation, reward, done

Use HorizonIDFDRslRlVecEnvWrapper for RSL-RL PPO training.
"""
from __future__ import annotations
import numpy as np, torch
import gymnasium as gym
from isaaclab.envs import DirectRLEnv
from pathlib import Path
import os
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
try:
    from kevin_integration.new_observations_pipeline.learned_models.velocity_to_valve_id import VelocityToValveMLP, load_finetuned_id_model
except ImportError:
    from finetune_id_physx_dynamic import VelocityToValveMLP
from kevin_integration.new_observations_pipeline.ppo1.observations_adapter import build_horizon_observation
from kevin_integration.new_observations_pipeline.ppo1.actions_adapter import ppo_action_to_horizon_displacement
from kevin_integration.new_observations_pipeline.ppo1.rewards_adapter import compute_horizon_reward
from kevin_integration.new_observations_pipeline.low_level_controller.idfd_rollout import closed_loop_idfd_rollout
from kevin_integration.new_observations_pipeline.low_level_controller.horizon_idfd_controller import compute_horizon_command


# Observation dimensions
EE_POSE_DIM = 7       # x,y,z,qw,qx,qy,qz
EE_ERROR_DIM = 4      # pos_err[3], orient_err[1]
JOINT_DIM = 10        # pos[5] + vel[5]
CYL_DIM = 8           # ext[4] + extvel[4]
HORIZON_CTX_DIM = 16  # prev valve[4], prev fd_delta[4], prev meas[4], prev cmd[4]
ACTION_CTX_DIM = 8    # prev action[4], prev action_rate[4]
VEHICLE_DIM = 7       # unimog pose[4] + vel[3]

HORIZON_OBS_DIM = (EE_POSE_DIM + EE_ERROR_DIM + JOINT_DIM + CYL_DIM +
                   HORIZON_CTX_DIM + ACTION_CTX_DIM)


class HorizonIDFDEnv:
    """Horizon-rate wrapper for MulagUpperLevelEnv with ID/FD pipeline.

    Designed to be wrapped by HorizonIDFDRslRlVecEnvWrapper.
    Exposes DirectRLEnv-compatible interface: observation_space, action_space, step(), reset().
    """
    def __init__(self, base_cfg: MulagUpperLevelEnvCfg, horizon_steps: int = 10,
                 action_scale_m: float = 0.01, fd_gains: list[float] | None = None,
                 device: str = "cuda:0"):
        self._H = horizon_steps
        self._action_scale = action_scale_m
        self._fd_gains = torch.tensor(fd_gains or [2.0, 3.0, 3.0, 3.0],
                                      dtype=torch.float32)
        self._device = device

        # Build base env
        base_cfg.low_level_servo_mode = "none"
        base_cfg.enable_scripted_vehicle_motion = False
        base_cfg.enable_startup_summary = True
        base_cfg.log_target_delta_xdot_clipping = False
        base_cfg.id_checkpoint_override_path = str(
            Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt"
        )
        apply_kevin_sim_memory_optimizations(base_cfg.sim)

        self._env = MulagUpperLevelEnv(cfg=base_cfg)
        self._uw = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        horizon_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self._uw.single_action_space = horizon_action_space
        self._uw.action_space = gym.vector.utils.batch_space(horizon_action_space, self._uw.num_envs)
        self._dt = float(self._uw.step_dt)
        self._model_dt = float(getattr(self._uw, "_sequence_model_dt", 0.05))

        # Load fine-tuned ID model
        id_path = Path(base_cfg.id_checkpoint_override_path).resolve()
        id_ckpt = torch.load(str(id_path), map_location=self._device, weights_only=False)
        self._id_model = VelocityToValveMLP().to(self._device)
        self._id_model.load_state_dict(id_ckpt["model_state_dict"])
        self._id_model.eval()
        self._X_mean = torch.from_numpy(id_ckpt["X_mean"]).to(self._device)
        self._X_std = torch.from_numpy(id_ckpt["X_std"]).to(self._device)

        # RSL-RL compatibility
        self.num_envs = self._env.num_envs
        self.device = self._device
        self.action_space = self._uw.action_space
        self.single_action_space = horizon_action_space
        self.num_actions = 4
        self.observation_space = HORIZON_OBS_DIM
        self.state_space = 0
        self.cfg = base_cfg

        # Per-env state for horizon context
        B = self.num_envs
        self._prev_action = torch.zeros((B, 4), device=self._device)
        self._prev_valve_cmd = torch.zeros((B, 4), device=self._device)
        self._prev_fd_delta = torch.zeros((B, 4), device=self._device)
        self._prev_meas_delta = torch.zeros((B, 4), device=self._device)
        self._prev_cmd_delta = torch.zeros((B, 4), device=self._device)
        self._horizon_reward = torch.zeros(B, device=self._device)
        self._horizon_terminated = torch.zeros(B, dtype=torch.bool, device=self._device)
        self._horizon_truncated = torch.zeros(B, dtype=torch.bool, device=self._device)

        print(f"[HorizonIDFDEnv] H={horizon_steps} action_scale={action_scale_m}m "
              f"fd_gains={self._fd_gains.tolist()} obs_dim={HORIZON_OBS_DIM} act_dim=4", flush=True)

    def _idfd_rollout(self, delta_x_des_H: torch.Tensor, x0: torch.Tensor, xd0: torch.Tensor) -> torch.Tensor:
        """Closed-loop ID/FD rollout, returns accumulated delta_fd_H."""
        id_hist = self._uw._sequence_id._step_history.detach().clone()
        fd_hist = self._uw._sequence_fd._step_history.detach().clone()
        x_r = x0.detach().clone()
        xd_r = xd0.detach().clone()
        fd_sum = torch.zeros((self.num_envs, 4), device=self._device)

        for _ in range(self._H):
            dx_step = delta_x_des_H / self._H
            v_step = dx_step / self._model_dt
            id_in = torch.cat([dx_step, v_step, x_r, xd_r], dim=-1)
            with torch.no_grad():
                valve = self._id_model((id_in - self._X_mean) / self._X_std)
            self._uw._sequence_fd.push_current_state(valve, x_r, xd_r)
            fd_d = self._uw._sequence_fd.predict_delta_extension().delta_extension_raw
            fd_sum += fd_d
            x_r += fd_d
            xd_r[:, :] = v_step

        self._uw._sequence_id._step_history = id_hist
        self._uw._sequence_fd._step_history = fd_hist
        return fd_sum

    def _smooth_ramp(self, delta_cmd_H: torch.Tensor, cq0: torch.Tensor):
        """Execute smooth position ramp over H steps."""
        dx_isaac = self._uw._cylinder_adapter.simscape_delta_to_isaac_delta(delta_cmd_H)
        lo = self._uw.robot.data.soft_joint_pos_limits[:, self._uw._cylinder_joint_ids, 0]
        hi = self._uw.robot.data.soft_joint_pos_limits[:, self._uw._cylinder_joint_ids, 1]
        za = torch.zeros((self.num_envs, 5), device=self._device)
        for i in range(1, self._H + 1):
            frac = i / self._H
            target = cq0 + frac * dx_isaac
            target = torch.clamp(target, lo, hi)
            self._uw._debug_manual_cylinder_position_target = target.detach().clone()
            obs, _, _, _, _ = self._env.step(za)
        self._uw._debug_manual_cylinder_position_target = None

    def _build_observation(self) -> torch.Tensor:
        """Build horizon-level observation vector."""
        ee_state = self._uw.robot.data.body_link_state_w[:, self._uw._ee_body_id, :7]
        # Normalize quaternion
        ee_quat = ee_state[:, 3:7]
        ee_quat = ee_quat / (ee_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        ee_pose = torch.cat([ee_state[:, :3], ee_quat], dim=-1)

        pos_err = self._uw._command_pos_w[:, :3] - ee_pose[:, :3]  # approximate
        orient_err = torch.zeros((self.num_envs, 1), device=self._device)  # placeholder

        jpos = self._uw.robot.data.joint_pos[:, self._uw._arm_joint_ids]
        jvel = self._uw.robot.data.joint_vel[:, self._uw._arm_joint_ids]
        cq = self._uw.robot.data.joint_pos[:, self._uw._cylinder_joint_ids]
        cqd = self._uw.robot.data.joint_vel[:, self._uw._cylinder_joint_ids]

        horizon_ctx = torch.cat([
            self._prev_valve_cmd, self._prev_fd_delta,
            self._prev_meas_delta, self._prev_cmd_delta
        ], dim=-1)

        action_ctx = torch.cat([
            self._prev_action,
            self._prev_action * 0.0  # placeholder for action rate
        ], dim=-1)

        return torch.cat([ee_pose, pos_err, orient_err, jpos, jvel, cq, cqd,
                          horizon_ctx, action_ctx], dim=-1)

    def reset(self):
        obs, _ = self._env.reset()
        self._prev_action.zero_()
        self._prev_valve_cmd.zero_()
        self._prev_fd_delta.zero_()
        self._prev_meas_delta.zero_()
        self._prev_cmd_delta.zero_()
        return self._build_observation(), {}

    def step(self, action: torch.Tensor):
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = torch.clamp(action, -1.0, 1.0)
        delta_x_des_H = action * self._action_scale

        # Pre-horizon state
        cq0 = self._uw.robot.data.joint_pos[:, self._uw._cylinder_joint_ids].detach().clone()
        x0, xd0 = self._uw._cylinder_adapter.current_from_cylinder_joints(
            cq0, self._uw.robot.data.joint_vel[:, self._uw._cylinder_joint_ids])

        # ID/FD rollout
        delta_fd_H = self._idfd_rollout(delta_x_des_H, x0, xd0)

        # Deployment command
        fd_gains = self._fd_gains.to(self._device).reshape(1, 4)
        delta_cmd_H = fd_gains * delta_fd_H
        delta_cmd_H = torch.clamp(delta_cmd_H, -0.02, 0.02)

        # Smooth ramp execution
        self._smooth_ramp(delta_cmd_H, cq0)

        # Post-horizon measurement
        cq1 = self._uw.robot.data.joint_pos[:, self._uw._cylinder_joint_ids].detach().clone()
        x1, _ = self._uw._cylinder_adapter.current_from_cylinder_joints(
            cq1, self._uw.robot.data.joint_vel[:, self._uw._cylinder_joint_ids])
        delta_x_meas_H = x1 - x0

        # Update horizon context
        self._prev_action = action.detach().clone()
        self._prev_fd_delta = delta_fd_H.detach().clone()
        self._prev_meas_delta = delta_x_meas_H.detach().clone()
        self._prev_cmd_delta = delta_cmd_H.detach().clone()

        # Horizon-level reward
        progress = torch.zeros(self.num_envs, device=self._device)
        alive_bonus = torch.ones(self.num_envs, device=self._device) * 0.01
        action_penalty = -0.01 * (action ** 2).sum(dim=-1)
        clip_penalty = torch.zeros(self.num_envs, device=self._device)
        reward = progress + alive_bonus + action_penalty + clip_penalty

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)

        return self._build_observation(), reward, terminated, truncated, {}

    @property
    def unwrapped(self):
        return self._env

    @property
    def max_episode_length(self):
        return self._env.max_episode_length

    def close(self):
        self._env.close()


class HorizonIDFDRslRlVecEnvWrapper(VecEnv):
    """RSL-RL adapter that keeps PPO on the horizon-level observation/action contract."""

    def __init__(self, env: HorizonIDFDEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.num_actions
        self.cfg = env.cfg
        self.env.reset()

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env._env.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.env._env.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = self.env.reset()
        return self._to_tensordict(obs), extras

    def get_observations(self) -> TensorDict:
        return self._to_tensordict(self.env._build_observation())

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self._to_tensordict(obs), rewards, dones, extras

    def close(self) -> None:
        self.env.close()

    def _to_tensordict(self, obs: torch.Tensor) -> TensorDict:
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=obs.device)
