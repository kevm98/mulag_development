"""
Stage-1 Adapter Environment Wrapper: Hierarchical control with frozen task-to-cylinder adapter.

Architecture:
    PPO action [ΔY_des, ΔZ_des]
    → learned adapter (frozen)
    → cylinder target [Δcyl₁, Δcyl₂, Δcyl₃, Δcyl₄]
    → lower-level cylinder controller
    → PhysX

PPO observation:
    [target_error_y, target_error_z, target_velocity_y, target_velocity_z,
     previous_ppo_action_y, previous_ppo_action_z]

PPO action:
    [desired_delta_ee_y, desired_delta_ee_z]
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import os

from rsl_rl.env import VecEnv

from kevin_integration.new_observations_pipeline.ppo1_adapter.task_to_cylinder_adapter import (
    TaskToCylinderAdapter,
)
from kevin_integration.new_observations_pipeline.ppo1_adapter.forward_response_model import (
    ForwardResponseModel,
)

SUCCESS_TOLERANCE = 0.10
OBS_DIM = 6  # [target_error_yz(2), target_vel_yz(2), prev_action(2)]


class Stage1AdapterEnvWrapper(VecEnv):
    """Stage-1 adapter wrapper with frozen learned task-to-cylinder adapter."""

    def __init__(
        self,
        cfg,
        adapter_checkpoint_path: str,
        forward_checkpoint_path: str | None = None,
        horizon_steps: int = 10,
        ppo_action_scale: float = 0.02,
        device: str = "cuda:0",
        kp_scale: float | None = None,
        kd_scale: float | None = None,
        force_rate_limit: float | None = None,
        velocity_filter_beta: float | None = None,
        trajectory_mode: str = "mixed",
        target_hold_steps: int = 30,
        post_reset_settle_steps: int = 30,
        backend: str = "explicit_effort_pid_bias",
        bias_calibration_time_s: float = 2.0,
        debug_bias_shapes: bool = False,
        reward_mode: str = "default",
        action_penalty_weight: float = 0.005,
        ee_jump_penalty_weight: float = 0.05,
        action_smoothing_beta: float = 0.0,
        action_rate_limit: float = -1.0,
        adapter_max_delta_cylinder: float = 0.0025,
        use_forward_model_validation: bool = False,
        debug_logging: bool = False,
    ):
        """
        Args:
            adapter_checkpoint_path: Path to trained adapter .pt file
            forward_checkpoint_path: Optional path to forward model (for validation)
            horizon_steps: Episode horizon
            ppo_action_scale: Scale for PPO output (desired EE motion)
            adapter_max_delta_cylinder: Max cylinder increment from adapter
            use_forward_model_validation: Whether to validate adapter outputs with forward model
            debug_logging: Whether to print debug logs for command flow
        """
        self._debug_logging = debug_logging
        # Configure environment
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
        cfg.explicit_force_rate_limit_fraction = (
            float(force_rate_limit) if force_rate_limit is not None else 0.01
        )
        cfg.explicit_pd_velocity_filter_beta = (
            float(velocity_filter_beta) if velocity_filter_beta is not None else 0.8
        )

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

        # Import environment
        from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import (
            MulagUpperLevelEnv,
        )
        from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

        # Set ID checkpoint (optional, can be bypassed)
        id_path = (
            Path(os.getcwd())
            / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt"
        ).resolve()
        if id_path.exists():
            cfg.id_checkpoint_override_path = str(id_path)

        apply_kevin_sim_memory_optimizations(cfg.sim)

        self.cfg = cfg
        self._env = MulagUpperLevelEnv(cfg=cfg)
        self._uw = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        self._H = int(horizon_steps)
        self._ppo_action_scale = float(ppo_action_scale)
        self._adapter_max_delta_cylinder = float(adapter_max_delta_cylinder)
        self._use_forward_validation = bool(use_forward_model_validation)
        self._device = device
        self._cyl_ids = self._uw._cylinder_joint_ids
        self._tool_id = self._uw._tool_joint_id
        self._ee_id = self._uw._ee_body_id
        self._dt = float(self._uw.step_dt)

        B = self.num_envs = self._uw.num_envs
        self.device = device
        self.max_episode_length = 200
        self.num_actions = 2  # PPO outputs [ΔY_des, ΔZ_des]
        self.num_obs = OBS_DIM

        self._trajectory_mode = trajectory_mode
        self._target_hold_steps = int(target_hold_steps)
        self._post_reset_settle_steps = int(post_reset_settle_steps)
        self._target_y_range = 0.05
        self._target_z_range = 0.02

        # Reward config
        self._reward_mode = reward_mode
        self._action_penalty_weight = action_penalty_weight
        self._ee_jump_penalty_weight = ee_jump_penalty_weight
        self._action_smoothing_beta = action_smoothing_beta
        self._action_rate_limit = action_rate_limit

        # Load frozen adapter (auto-detect type)
        checkpoint = torch.load(adapter_checkpoint_path, map_location=device)
        adapter_type = checkpoint.get("type", "learned")  # default to learned for backward compat

        if adapter_type == "analytical":
            from kevin_integration.new_observations_pipeline.ppo1_adapter.analytical_adapter import AnalyticalAdapter
            self._adapter = AnalyticalAdapter.load(adapter_checkpoint_path, device=device)
            print(f"[Stage1AdapterEnvWrapper] Loaded ANALYTICAL adapter from {adapter_checkpoint_path}")
        else:
            self._adapter = TaskToCylinderAdapter.load(adapter_checkpoint_path, device=device)
            print(f"[Stage1AdapterEnvWrapper] Loaded LEARNED adapter from {adapter_checkpoint_path}")

        self._adapter.eval()
        for param in self._adapter.parameters():
            param.requires_grad = False

        # Optionally load forward model for validation
        self._forward_model = None
        if forward_checkpoint_path and Path(forward_checkpoint_path).exists():
            self._forward_model = ForwardResponseModel.load(forward_checkpoint_path, device=device)
            self._forward_model.eval()
            for param in self._forward_model.parameters():
                param.requires_grad = False
            print(f"[Stage1AdapterEnvWrapper] Loaded forward model from {forward_checkpoint_path}")

        # State buffers
        self._target_pos_yz = torch.zeros((B, 2), device=device)
        self._target_vel_yz = torch.zeros((B, 2), device=device)
        self._prev_ee_yz = torch.zeros((B, 2), device=device)
        self._current_ee_yz = torch.zeros((B, 2), device=device)
        self._prev_ppo_action = torch.zeros((B, 2), device=device)
        self._current_ppo_action = torch.zeros((B, 2), device=device)
        self._prev_cylinder_delta = torch.zeros((B, 4), device=device)
        self._current_cylinder_delta = torch.zeros((B, 4), device=device)

        # Episode state
        self._episode_step = torch.zeros(B, dtype=torch.long, device=device)
        self._target_change_counter = torch.zeros(B, dtype=torch.long, device=device)
        self._success_count = torch.zeros(B, dtype=torch.long, device=device)

        # Metrics
        self._total_distance_error = torch.zeros(B, device=device)
        self._total_ee_jump = torch.zeros(B, device=device)
        self._cylinder_limit_violations = torch.zeros(B, dtype=torch.long, device=device)

        self._resample_targets()

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Get PPO observation: target error, target velocity, previous action."""
        # Get current EE position
        ee_body = self._uw.scene["robot"].data.body_state_w[:, self._ee_id, :]
        ee_pos_w = ee_body[:, :3]
        root_body = self._uw.scene["robot"].data.root_state_w
        root_pos_w = root_body[:, :3]
        root_quat_w = root_body[:, 3:7]

        from isaaclab.utils.math import quat_apply_inverse
        ee_pos_rel = quat_apply_inverse(root_quat_w, ee_pos_w - root_pos_w)
        self._current_ee_yz = ee_pos_rel[:, [1, 2]]  # Y and Z in root frame

        # Compute target error
        target_error = self._target_pos_yz - self._current_ee_yz

        # Stack observation
        obs = torch.cat([
            target_error,  # (2)
            self._target_vel_yz,  # (2)
            self._prev_ppo_action,  # (2)
        ], dim=-1)  # Total: 6D

        extras = {
            "observations": {"policy": obs},
            "current_ee_yz": self._current_ee_yz,
            "target_pos_yz": self._target_pos_yz,
            "target_error": target_error,
        }

        return obs, extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset environment and resample targets."""
        obs_dict = self._env.reset()

        # Reset state
        self._episode_step[:] = 0
        self._target_change_counter[:] = 0
        self._success_count[:] = 0
        self._total_distance_error[:] = 0.0
        self._total_ee_jump[:] = 0.0
        self._cylinder_limit_violations[:] = 0
        self._prev_ppo_action[:] = 0.0
        self._current_ppo_action[:] = 0.0
        self._prev_cylinder_delta[:] = 0.0
        self._current_cylinder_delta[:] = 0.0

        # Resample all targets
        self._resample_targets()

        # Settle for calibration/bias
        for _ in range(self._post_reset_settle_steps):
            zero_action = torch.zeros((self.num_envs, 5), device=self.device)
            self._env.step(zero_action)

        # Get initial observation
        obs, extras = self.get_observations()
        self._prev_ee_yz = self._current_ee_yz.clone()

        return obs, extras

    def step(self, ppo_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Execute one step with PPO action.

        Args:
            ppo_action: (B, 2) tensor [desired_delta_y, desired_delta_z]

        Returns:
            obs, rewards, dones, extras
        """
        # Scale PPO action to desired EE motion
        desired_delta_ee_yz = ppo_action * self._ppo_action_scale

        # Action smoothing (optional)
        if self._action_smoothing_beta > 0:
            desired_delta_ee_yz = (
                self._action_smoothing_beta * self._prev_ppo_action
                + (1 - self._action_smoothing_beta) * desired_delta_ee_yz
            )

        # Action rate limiting (optional)
        if self._action_rate_limit > 0:
            delta_action = desired_delta_ee_yz - self._prev_ppo_action
            delta_action = torch.clamp(
                delta_action,
                -self._action_rate_limit,
                self._action_rate_limit,
            )
            desired_delta_ee_yz = self._prev_ppo_action + delta_action

        self._current_ppo_action = desired_delta_ee_yz.clone()

        # Get current cylinder state
        cylinder_pos = self._uw.scene["robot"].data.joint_pos[:, self._cyl_ids]
        cylinder_vel = self._uw.scene["robot"].data.joint_vel[:, self._cyl_ids]

        # Run adapter to get cylinder command
        with torch.no_grad():
            cylinder_delta = self._adapter(
                desired_delta_ee_yz,
                self._current_ee_yz,
                cylinder_pos,
                cylinder_vel,
                self._prev_cylinder_delta,
                enforce_limits=True,
            )

        # Optional: Validate with forward model
        if self._use_forward_validation and self._forward_model is not None:
            with torch.no_grad():
                predicted_ee_delta = self._forward_model(
                    self._current_ee_yz,
                    cylinder_pos,
                    cylinder_vel,
                    self._prev_cylinder_delta,
                    cylinder_delta,
                )
                # Could add logic here to reject commands with bad forward prediction

        self._current_cylinder_delta = cylinder_delta.clone()

        # Convert cylinder delta to cylinder target
        cylinder_target = cylinder_pos + cylinder_delta

        # Check limit violations
        limit_margin = 0.01
        near_limit = (
            (cylinder_target < self._adapter.cylinder_min + limit_margin).any(dim=1)
            | (cylinder_target > self._adapter.cylinder_max - limit_margin).any(dim=1)
        )
        self._cylinder_limit_violations += near_limit.long()

        # Send cylinder target to lower-level controller
        # The underlying env expects 5D action: [cyl1, cyl2, cyl3, cyl4, tool]
        # We control cylinders only; tool is passive
        tool_target = torch.zeros((self.num_envs, 1), device=self.device)
        low_level_action = torch.cat([cylinder_target, tool_target], dim=-1)

        # DEBUG: Log first environment's command flow (set DEBUG_LOGGING=True to enable)
        if hasattr(self, '_debug_logging') and self._debug_logging and self._step_counter % 10 == 0:
            env_idx = 0
            print(f"\n[WRAPPER DEBUG] Step {self._step_counter}, Env {env_idx}:")
            print(f"  PPO action (scaled): [{action[env_idx,0].item()*1000:+6.2f}, {action[env_idx,1].item()*1000:+6.2f}] mm")
            print(f"  Desired EE delta: [{desired_delta_ee_yz[env_idx,0].item()*1000:+6.2f}, {desired_delta_ee_yz[env_idx,1].item()*1000:+6.2f}] mm")
            print(f"  Current cyl pos: [{cylinder_pos[env_idx,0].item()*1000:+6.1f}, {cylinder_pos[env_idx,1].item()*1000:+6.1f}, {cylinder_pos[env_idx,2].item()*1000:+6.1f}, {cylinder_pos[env_idx,3].item()*1000:+6.1f}] mm")
            print(f"  Adapter output (delta): [{cylinder_delta[env_idx,0].item()*1000:+6.2f}, {cylinder_delta[env_idx,1].item()*1000:+6.2f}, {cylinder_delta[env_idx,2].item()*1000:+6.2f}, {cylinder_delta[env_idx,3].item()*1000:+6.2f}] mm")
            print(f"  Cylinder target: [{cylinder_target[env_idx,0].item()*1000:+6.1f}, {cylinder_target[env_idx,1].item()*1000:+6.1f}, {cylinder_target[env_idx,2].item()*1000:+6.1f}, {cylinder_target[env_idx,3].item()*1000:+6.1f}] mm")
            print(f"  Cylinder IDs: {self._cyl_ids}")

        # Execute in Isaac environment (DirectRLEnv returns 5 values)
        obs_dict, _, terminated, truncated, info_dict = self._env.step(low_level_action)

        # Get new observation
        obs, extras = self.get_observations()

        # Compute EE jump
        ee_jump = (self._current_ee_yz - self._prev_ee_yz).norm(dim=-1)
        self._total_ee_jump += ee_jump

        # Compute distance to target
        distance = (self._current_ee_yz - self._target_pos_yz).norm(dim=-1)
        self._total_distance_error += distance

        # Check success
        success = distance < SUCCESS_TOLERANCE
        self._success_count += success.long()

        # Compute reward
        rewards = self._compute_reward(
            distance,
            ee_jump,
            desired_delta_ee_yz,
            cylinder_delta,
        )

        # Update state
        self._episode_step += 1
        self._target_change_counter += 1
        self._prev_ee_yz = self._current_ee_yz.clone()
        self._prev_ppo_action = self._current_ppo_action.clone()
        self._prev_cylinder_delta = self._current_cylinder_delta.clone()

        # Check if target should change
        target_timeout = self._target_change_counter >= self._target_hold_steps
        if target_timeout.any():
            self._resample_targets(mask=target_timeout)

        # Check episode termination
        dones = self._episode_step >= self.max_episode_length

        # Collect extras
        extras.update({
            "episode": {
                "mean_distance": self._total_distance_error / (self._episode_step + 1),
                "success_rate": self._success_count.float() / (self._episode_step + 1),
                "mean_ee_jump": self._total_ee_jump / (self._episode_step + 1),
                "cylinder_limit_violations": self._cylinder_limit_violations.float(),
                "ppo_action_norm": ppo_action.abs().mean(),
                "adapter_output_norm": cylinder_delta.abs().mean(),
            },
            "distance": distance,
            "ee_jump": ee_jump,
            "ppo_action": desired_delta_ee_yz,
            "adapter_output": cylinder_delta,
            "cylinder_target": cylinder_target,
        })

        return obs, rewards, dones, extras

    def _compute_reward(
        self,
        distance: torch.Tensor,
        ee_jump: torch.Tensor,
        ppo_action: torch.Tensor,
        adapter_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward based on distance, jump penalty, and action penalty."""
        # Primary reward: negative distance
        reward_distance = -distance

        # Penalties
        penalty_action = -self._action_penalty_weight * ppo_action.abs().sum(dim=-1)
        penalty_jump = -self._ee_jump_penalty_weight * ee_jump

        if self._reward_mode == "distance_only":
            return reward_distance
        else:
            return reward_distance + penalty_action + penalty_jump

    def _resample_targets(self, mask: torch.Tensor | None = None):
        """Resample target positions with optional mask."""
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        N = mask.sum().item()
        if N == 0:
            return

        # Sample new targets
        if self._trajectory_mode == "sequence_easy":
            # Simple Y-Z sequences
            new_y = torch.randn(N, device=self.device) * self._target_y_range
            new_z = torch.randn(N, device=self.device) * self._target_z_range + 0.5  # Elevated
            new_targets = torch.stack([new_y, new_z], dim=-1)
            new_vels = torch.zeros((N, 2), device=self.device)
        else:
            # Random targets
            new_y = torch.randn(N, device=self.device) * self._target_y_range
            new_z = torch.randn(N, device=self.device) * self._target_z_range + 0.5
            new_targets = torch.stack([new_y, new_z], dim=-1)
            new_vels = torch.zeros((N, 2), device=self.device)

        # Update targets
        self._target_pos_yz[mask] = new_targets
        self._target_vel_yz[mask] = new_vels
        self._target_change_counter[mask] = 0

    def close(self):
        """Close environment."""
        self._env.close()
