from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from kevin_integration.rl import AdaptiveRLLearnedController, AdaptiveRLLearnedControllerCfg, build_rl_obs
from kevin_integration.rl.action_adapter import clamp_and_sanitize
from kevin_integration.rl.reward_terms import JointReachRewardWeights, joint_reach_reward, joint_safety_termination
from kevin_integration.tasks.mulag_adaptive_rl_env_cfg import MulagAdaptiveRLJointReachEnvCfg


class MulagAdaptiveRLEnv(DirectRLEnv):
    cfg: MulagAdaptiveRLJointReachEnvCfg

    def __init__(self, cfg: MulagAdaptiveRLJointReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._state_joint_ids = self._find_joint_ids(self.cfg.state_joint_names)
        self._command_joint_ids = self._find_joint_ids(self.cfg.command_joint_names)
        if len(self._state_joint_ids) != 5:
            raise RuntimeError(f"Expected 5 state joints, got {len(self._state_joint_ids)}")
        if len(self._command_joint_ids) != 4:
            raise RuntimeError(f"Expected 4 command joints, got {len(self._command_joint_ids)}")

        self._controller = AdaptiveRLLearnedController(
            AdaptiveRLLearnedControllerCfg(
                models_dir=self.cfg.models_dir,
                num_envs=self.num_envs,
                device=self.device,
                fallback_window=self.cfg.fallback_window,
                max_abs_valve=self.cfg.max_abs_valve,
                max_abs_torque=self.cfg.max_abs_torque,
                max_abs_pressure_delta=self.cfg.max_abs_pressure_delta,
                max_abs_fnet=self.cfg.max_abs_fnet,
                sd_output_mode=self.cfg.sd_output_mode,
            )
        )
        self._reward_weights = JointReachRewardWeights(
            q_tracking=self.cfg.rew_q_tracking,
            qdot=self.cfg.rew_qdot,
            action_smoothness=self.cfg.rew_action_smoothness,
            valve_effort=self.cfg.rew_valve_effort,
            torque_effort=self.cfg.rew_torque_effort,
            alive=self.cfg.rew_alive,
            termination=self.cfg.rew_termination,
        )

        self._q_ref = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self._qdot_ref = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self._dP = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._fnet = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._prev_action = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._current_action = torch.zeros_like(self._prev_action)
        self._action_delta = torch.zeros_like(self._prev_action)
        self._prev_valve_cmd = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._current_valve_cmd = torch.zeros_like(self._prev_valve_cmd)
        self._current_torque = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._z_arm_hat = torch.zeros((self.num_envs, 24), dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _find_joint_ids(self, joint_names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for joint_name in joint_names:
            joint_ids, _ = self.robot.find_joints(joint_name)
            if len(joint_ids) == 0:
                raise RuntimeError(f"Could not find joint: {joint_name}")
            ids.append(joint_ids[0])
        return ids

    def _read_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.robot.data.joint_pos[:, self._state_joint_ids]
        qdot = self.robot.data.joint_vel[:, self._state_joint_ids]
        return q, qdot

    def _read_pressure_delta(self) -> torch.Tensor:
        if self.cfg.pressure_noise_std <= 0.0:
            return self._dP
        return self._dP + self.cfg.pressure_noise_std * torch.randn_like(self._dP)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._current_action = self.cfg.action_scale * clamp_and_sanitize(actions, 1.0)
        self._action_delta = self._current_action - self._prev_action

        q, qdot = self._read_joint_state()
        dP = self._read_pressure_delta()
        control_out = self._controller.apply_action(
            q,
            qdot,
            dP,
            self._fnet,
            self._current_action,
            z_arm_hat=self._z_arm_hat,
        )
        self._current_torque = control_out.torque
        self._current_valve_cmd = control_out.valve_cmd
        self._z_arm_hat = control_out.z_arm_hat
        self._dP = control_out.dP_next
        self._fnet = control_out.fnet_next

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self._current_torque, joint_ids=self._command_joint_ids)

    def _get_observations(self) -> dict:
        q, qdot = self._read_joint_state()
        dP = self._read_pressure_delta()
        self._z_arm_hat = self._controller.compute_z_arm(q, qdot, dP, self._prev_valve_cmd)
        obs = build_rl_obs(
            q, qdot, self._q_ref, self._qdot_ref, self._prev_action, self._prev_valve_cmd, self._z_arm_hat
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        q, qdot = self._read_joint_state()
        reward = joint_reach_reward(
            q,
            qdot,
            self._q_ref,
            self._action_delta,
            self._current_valve_cmd,
            self._current_torque,
            self.reset_terminated,
            self._reward_weights,
        )
        self._prev_action = self._current_action.clone()
        self._prev_valve_cmd = self._current_valve_cmd.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        q, qdot = self._read_joint_state()
        lower_limits = self.robot.data.soft_joint_pos_limits[:, self._state_joint_ids, 0]
        upper_limits = self.robot.data.soft_joint_pos_limits[:, self._state_joint_ids, 1]
        safety_terminated = joint_safety_termination(
            q,
            qdot,
            lower_limits,
            upper_limits,
            joint_limit_margin=self.cfg.joint_limit_margin,
            max_abs_joint_velocity=self.cfg.max_abs_joint_velocity,
        )
        finite_control = torch.isfinite(self._current_torque).all(dim=-1) & torch.isfinite(
            self._current_valve_cmd
        ).all(dim=-1)
        terminated = safety_terminated | (~finite_control)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_root_pose, default_root_velocity = self._default_root_state(env_ids)
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]

        self._sample_joint_targets(env_ids)
        self._qdot_ref[env_ids] = 0.0

        self._dP[env_ids] = 0.0
        self._fnet[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._current_action[env_ids] = 0.0
        self._action_delta[env_ids] = 0.0
        self._prev_valve_cmd[env_ids] = 0.0
        self._current_valve_cmd[env_ids] = 0.0
        self._current_torque[env_ids] = 0.0
        self._z_arm_hat[env_ids] = 0.0
        self._controller.reset(env_ids)

        self.robot.write_root_pose_to_sim(default_root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(default_root_velocity, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _default_root_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.robot.data, "default_root_pose") and hasattr(self.robot.data, "default_root_vel"):
            root_pose = self.robot.data.default_root_pose[env_ids].clone()
            root_velocity = self.robot.data.default_root_vel[env_ids].clone()
            return root_pose, root_velocity

        root_state = self.robot.data.default_root_state[env_ids].clone()
        return root_state[:, :7], root_state[:, 7:]

    def _sample_joint_targets(self, env_ids: torch.Tensor) -> None:
        default_q = self.robot.data.default_joint_pos[env_ids][:, self._state_joint_ids]
        lower = self.robot.data.soft_joint_pos_limits[env_ids][:, self._state_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[env_ids][:, self._state_joint_ids, 1]
        target_range = torch.tensor(self.cfg.target_joint_range, dtype=torch.float32, device=self.device).reshape(1, 5)

        has_limits = torch.isfinite(lower) & torch.isfinite(upper)
        has_limits = has_limits & ((upper - lower) > 2.0 * self.cfg.joint_limit_margin)
        fallback_lower = default_q - target_range
        fallback_upper = default_q + target_range
        safe_lower = torch.where(has_limits, lower + self.cfg.joint_limit_margin, fallback_lower)
        safe_upper = torch.where(has_limits, upper - self.cfg.joint_limit_margin, fallback_upper)
        sample_lower = torch.maximum(fallback_lower, safe_lower)
        sample_upper = torch.minimum(fallback_upper, safe_upper)
        sample_upper = torch.maximum(sample_upper, sample_lower + 1.0e-3)

        self._q_ref[env_ids] = sample_lower + torch.rand_like(default_q) * (sample_upper - sample_lower)
