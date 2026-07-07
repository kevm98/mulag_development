#!/usr/bin/env python3
"""Horizon-rate PPO environment wrapper with ID/FD closed-loop rollout.

Each env.step(action) executes H physics steps:
    action[0:4] in [-1, 1] -> delta_x_des_H = action * action_scale_m
    -> closed-loop ID/FD rollout -> delta_fd_H
    -> delta_cmd_H = fd_gain[c] * delta_fd_H
    -> smooth PhysX ramp over H steps
    -> horizon-level observation, reward, done

Use HorizonIDFDRslRlVecEnvWrapper for RSL-RL PPO training.
"""
from __future__ import annotations

import os
import csv
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.observations_adapter import (
    HORIZON_OBS_DIM,
    build_high_level_observation,
    HI_HORIZON_OBS_DIM,
    build_horizon_observation,
)
from kevin_integration.new_observations_pipeline.ppo1.rewards_adapter import (
    compute_horizon_reward,
    compute_stage0_reward,
    compute_high_level_reward,
    compute_tracking_error,
)
from kevin_integration.new_observations_pipeline.low_level_controller.idfd_rollout import (
    closed_loop_idfd_rollout,
)
from kevin_integration.new_observations_pipeline.low_level_controller.high_level_ee_controller import (
    HighLevelEEController,
)
from kevin_integration.new_observations_pipeline.low_level_controller.horizon_idfd_controller import (
    compute_horizon_command,
)
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

try:
    from kevin_integration.new_observations_pipeline.learned_models.velocity_to_valve_id import (
        VelocityToValveMLP,
    )
except ImportError:
    from finetune_id_physx_dynamic import VelocityToValveMLP


class HorizonIDFDEnv:
    """Horizon-rate wrapper for MulagUpperLevelEnv with the selected ID/FD pipeline."""

    DONE_CAUSES = (
        "success",
        "timeout",
        "cab_guard_violation",
        "workspace_violation",
        "collision",
        "joint_limit",
        "cylinder_limit",
        "ground_safety",
        "NaN/nonfinite",
        "base_flying",
        "reset_hold_failure",
        "unknown_done",
    )

    def __init__(
        self,
        base_cfg: MulagUpperLevelEnvCfg,
        horizon_steps: int = 10,
        action_scale_m: float = 0.01,
        fd_gains: list[float] | None = None,
        device: str = "cuda:0",
    ):
        self._H = int(horizon_steps)
        self._action_scale = float(action_scale_m)
        self._fd_gains = torch.tensor(
            fd_gains or [2.0, 3.0, 3.0, 3.0],
            dtype=torch.float32,
            device=device,
        ).reshape(1, 4)
        # Override fd_gains for task_space mode (mapper handles scaling)
        if str(getattr(base_cfg, "ee_action_mode", "cylinder_space")) == "task_space_delta":
            self._fd_gains = torch.ones((1, 4), dtype=torch.float32, device=device)
        self._device = device

        base_cfg.low_level_servo_mode = "none"
        base_cfg.lowlevel_actuator_backend = "explicit_effort_pid_bias"
        base_cfg.enable_scripted_vehicle_motion = False
        base_cfg.horizon_pin_root_pose = True  # lock base for high-level PPO
        base_cfg.enable_startup_summary = True
        base_cfg.log_target_delta_xdot_clipping = False
        base_cfg.id_checkpoint_override_path = str(
            Path(os.getcwd())
            / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt"
        )
        apply_kevin_sim_memory_optimizations(base_cfg.sim)

        self._env = MulagUpperLevelEnv(cfg=base_cfg)
        self._uw = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        horizon_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self._uw.single_action_space = horizon_action_space
        self._uw.action_space = gym.vector.utils.batch_space(horizon_action_space, self._uw.num_envs)
        self._dt = float(self._uw.step_dt)
        self._model_dt = float(getattr(self._uw, "_sequence_model_dt", 0.05))

        id_path = Path(base_cfg.id_checkpoint_override_path).resolve()
        id_ckpt = torch.load(str(id_path), map_location=self._device, weights_only=False)
        self._id_model = VelocityToValveMLP().to(self._device)
        self._id_model.load_state_dict(id_ckpt["model_state_dict"])
        self._id_model.eval()
        self._X_mean = torch.from_numpy(id_ckpt["X_mean"]).to(self._device)
        self._X_std = torch.from_numpy(id_ckpt["X_std"]).to(self._device)

        self.num_envs = self._env.num_envs
        self.device = self._device

        # Determine action mode: task_space or cylinder_space
        self._ee_mode = str(getattr(base_cfg, "ee_action_mode", "cylinder_space"))
        self._is_task_space = self._ee_mode == "task_space_delta"

        if self._is_task_space:
            self._ee_action_scale = float(getattr(base_cfg, "ee_action_scale_m", 0.05))
            act_dim = int(getattr(base_cfg, "ee_action_dim", 3))
            self.observation_space = HI_HORIZON_OBS_DIM
        else:
            act_dim = 4
            self.observation_space = HORIZON_OBS_DIM

        horizon_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self._uw.single_action_space = horizon_action_space
        self._uw.action_space = gym.vector.utils.batch_space(horizon_action_space, self._uw.num_envs)
        self.action_space = self._uw.action_space
        self.single_action_space = horizon_action_space
        self.num_actions = act_dim
        self.state_space = 0
        self.cfg = base_cfg
        B = self.num_envs
        self._prev_action = torch.zeros((B, self.num_actions), device=self._device)
        # -- High-level EE controller (task_space mode) --
        self._ee_controller: HighLevelEEController | None = None
        self._prev_hl_action = torch.zeros((B, self.num_actions), device=self._device)
        self._jacobian_valid = torch.zeros(B, dtype=torch.bool, device=self._device)
        if self._is_task_space:
            self._ee_controller = HighLevelEEController(
                device=self._device,
                eps=float(getattr(base_cfg, "ee_jacobian_eps", 0.001)),
                k_p=float(getattr(base_cfg, "ee_k_p", 0.15)),
                lam=float(getattr(base_cfg, "ee_lam", 1e-2)),
                per_cyl_clip=float(getattr(base_cfg, "ee_per_cyl_clip_m", 0.0015)),
                global_clip=float(getattr(base_cfg, "ee_global_clip_m", 0.003)),
                smooth_alpha=float(getattr(base_cfg, "ee_smooth_alpha", 0.7)),
            )
        self._prev_delta_x_des = torch.zeros((B, 4), device=self._device)
        self._prev_fd_delta = torch.zeros((B, 4), device=self._device)
        self._prev_meas_delta = torch.zeros((B, 4), device=self._device)
        self._prev_cmd_delta = torch.zeros((B, 4), device=self._device)
        self._prev_guard_flags = torch.zeros((B, 4), device=self._device)
        self._cylinder_hold_target = torch.zeros((B, 4), device=self._device)
        self._tool_hold_target = torch.zeros((B, 1), device=self._device)
        self._root_hold_pose = torch.zeros((B, 7), device=self._device)
        self._root_hold_velocity = torch.zeros((B, 6), device=self._device)
        self._last_reward_terms: dict[str, torch.Tensor] = {}
        self._last_guard_fraction = torch.zeros(B, device=self._device)
        self._success_hold_steps = torch.zeros(B, dtype=torch.long, device=self._device)

        self._ws_consecutive_steps = torch.zeros(B, dtype=torch.long, device=self._device)
        self._diagnostic_dir = Path("outputs/cylinder_diagnostics/paper_matched_pipeline/ppo_training")
        self._diagnostic_csv = self._diagnostic_dir / "termination_diagnostics.csv"
        self._diagnostic_report = self._diagnostic_dir / "termination_diagnostics_report.md"
        self._analysis_report = self._diagnostic_dir / "ppo_iteration_termination_analysis.md"
        self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self._csv_header_written = False
        self._episode_records: list[dict[str, float | int | str]] = []
        self._episode_return = torch.zeros(B, device=self._device)
        self._horizon_episode_length = torch.zeros(B, dtype=torch.long, device=self._device)
        self._episode_ee_error_start = compute_tracking_error(self._uw).detach().clone()
        self._episode_guard_sum = torch.zeros(B, device=self._device)
        self._episode_clip_sum = torch.zeros(B, device=self._device)
        self._episode_hydraulic_sum = torch.zeros(B, device=self._device)
        self._episode_action_norm_sum = torch.zeros(B, device=self._device)
        self._episode_action_rate_sum = torch.zeros(B, device=self._device)
        self._episode_delta_x_des_sum = torch.zeros(B, device=self._device)
        self._episode_delta_x_meas_sum = torch.zeros(B, device=self._device)
        self._episode_min_cab_clearance = torch.full((B,), float("inf"), device=self._device)
        self._latest_log: dict[str, torch.Tensor] = {}
        self._trajectory_mode = str(getattr(base_cfg, "trajectory_mode", "none"))
        self._trajectory_plane = str(getattr(base_cfg, "trajectory_plane", "ground_xy"))
        self._ground_workspace_json = str(getattr(base_cfg, "ground_workspace_json", "outputs/ppo1_horizon_idfd/ground_workspace.json"))
        self._tool_contact_mode = str(getattr(base_cfg, "tool_contact_mode", "geometry_aabb"))
        self._tool_ground_clearance = float(getattr(base_cfg, "tool_ground_clearance", 0.03))
        self._min_arm_ground_clearance = float(getattr(base_cfg, "min_arm_ground_clearance", 0.02))
        self._ground_z = float(getattr(base_cfg, "ground_z", 0.0))
        self._trajectory_scale = float(getattr(base_cfg, "trajectory_scale", 0.8))
        self._ground_path_world: torch.Tensor | None = None
        self._ground_target_tool_bottom_world = torch.zeros((B, 3), device=self._device)
        self._ground_path_index = 0
        self._ground_path_completion = torch.zeros(B, device=self._device)
        self._ground_tracking_clearance = self._tool_ground_clearance
        self._ground_limitation = ""
        self._ground_path_params: dict[str, float | int | str | list] = {}
        self._ground_visuals_ready = False
        self._ground_visual_warning_printed = False
        self._ground_path_marker_paths: list[str] = []
        self._ground_current_marker_path = "/World/ground_trajectory/current_target"
        self._ground_ee_origin_marker_path = "/World/ground_trajectory/ee_origin"
        self._ground_tool_contact_marker_path = "/World/ground_trajectory/tool_contact"
        self._ground_target_contact_marker_path = "/World/ground_trajectory/target_tool_contact"
        self._ground_tool_aabb_marker_path = "/World/ground_trajectory/tool_aabb"
        self._ground_trail_marker_paths: list[str] = []
        self._ground_trail_points: list[tuple[float, float, float]] = []
        self._tool_contact_warning_printed = False

        self._refresh_root_hold_from_current()
        self._refresh_hold_target_from_current()
        print(
            f"[HorizonIDFDEnv] H={self._H} action_scale={self._action_scale}m "
            f"fd_gains={self._fd_gains.flatten().tolist()} obs_dim={self.observation_space} act_dim={act_dim}",
            flush=True,
        )

    def _joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        limits = self._uw.robot.data.soft_joint_pos_limits[:, self._uw._cylinder_joint_ids]
        margin = float(getattr(self.cfg, "horizon_cylinder_limit_margin", 0.002))
        lo = limits[..., 0] + margin
        hi = limits[..., 1] - margin
        return torch.minimum(lo, hi), torch.maximum(lo, hi)

    def _refresh_hold_target_from_current(self) -> None:
        cq = self._uw.robot.data.joint_pos[:, self._uw._cylinder_joint_ids].detach().clone()
        lo, hi = self._joint_limits()
        self._cylinder_hold_target = torch.clamp(cq, lo, hi)
        tool_q = self._uw.robot.data.joint_pos[:, self._uw._tool_joint_id].detach().clone().unsqueeze(-1)
        self._tool_hold_target = torch.nan_to_num(tool_q, nan=0.0, posinf=0.0, neginf=0.0)
        self._uw.set_debug_manual_cylinder_targets(
            self._cylinder_hold_target,
            tool_position_target=self._tool_hold_target,
        )

    def _refresh_root_hold_from_current(self) -> None:
        self._root_hold_pose = self._uw.robot.data.root_state_w[:, :7].detach().clone()
        self._root_hold_velocity = torch.zeros((self.num_envs, 6), device=self._device)

    def _pin_root_pose_if_enabled(self) -> None:
        if not bool(getattr(self.cfg, "horizon_pin_root_pose", True)):
            return
        self._uw.robot.write_root_pose_to_sim(self._root_hold_pose)
        self._uw.robot.write_root_velocity_to_sim(self._root_hold_velocity)

    def _settle_reset_hold(self) -> None:
        settle_steps = int(getattr(self.cfg, "horizon_reset_settle_steps", 10))
        if settle_steps <= 0:
            return
        zero_action = torch.zeros((self.num_envs, 5), device=self._device)
        for _ in range(settle_steps):
            self._uw.set_debug_manual_cylinder_targets(
                self._cylinder_hold_target,
                tool_position_target=self._tool_hold_target,
            )
            self._pin_root_pose_if_enabled()
            self._env.step(zero_action)
            self._pin_root_pose_if_enabled()
        self._env.episode_length_buf.zero_()
        self._refresh_root_hold_from_current()
        self._refresh_hold_target_from_current()

    def _tcp_pos_base_frame(self) -> torch.Tensor:
        ee_pos_w = self._uw.robot.data.body_link_state_w[:, self._uw._ee_body_id, :3]
        base_pos_w = self._uw.robot.data.root_pos_w
        base_quat_w = self._uw.robot.data.root_quat_w
        base_quat_w = base_quat_w / base_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)

    def get_tool_origin_world_pos(self) -> torch.Tensor:
        return self._uw.robot.data.body_link_state_w[:, self._uw._ee_body_id, :3]

    def _manual_tool_offset_world(self) -> torch.Tensor:
        ee_state = self._uw.robot.data.body_link_state_w[:, self._uw._ee_body_id, :7]
        ee_pos_w = ee_state[:, :3]
        x = float(getattr(self.cfg, "tool_tip_local_offset_x", 0.0))
        y = float(getattr(self.cfg, "tool_tip_local_offset_y", 0.0))
        z = float(getattr(self.cfg, "tool_tip_local_offset_z", 0.0))
        legacy_offset = getattr(self.cfg, "tool_tip_local_offset", (0.0, 0.0, 0.0))
        if (x, y, z) == (0.0, 0.0, 0.0) and legacy_offset is not None:
            x, y, z = (float(legacy_offset[0]), float(legacy_offset[1]), float(legacy_offset[2]))
        local_offset = torch.tensor(
            (x, y, z),
            dtype=ee_pos_w.dtype,
            device=self._device,
        ).reshape(1, 3)
        ee_quat_w = ee_state[:, 3:7]
        ee_quat_w = ee_quat_w / ee_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return quat_apply(ee_quat_w, local_offset.expand(self.num_envs, -1))

    def _ee_or_tool_world_pos(self) -> torch.Tensor:
        return self.get_tool_bottom_world_pos()

    def _effective_tool_ground_clearance(self) -> float:
        return float(getattr(self.cfg, "tool_ground_clearance", 0.03))

    def _robot_env_prim_path(self, env_id: int) -> str:
        return f"/World/envs/env_{int(env_id)}/Robot"

    def _find_body_prim_path(self, body_name: str, env_id: int = 0) -> str | None:
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            root_path = self._robot_env_prim_path(env_id)
            root = stage.GetPrimAtPath(root_path)
            if not root or not root.IsValid():
                return None
            fallback = None
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith(root_path + "/"):
                    continue
                name = prim.GetName()
                if name == body_name:
                    return path
                if fallback is None and body_name in name:
                    fallback = path
            return fallback
        except Exception:
            return None

    def _compute_prim_world_aabb(self, prim_path: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return None
            purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
            bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            mn = bbox.GetMin()
            mx = bbox.GetMax()
            min_t = torch.tensor([float(mn[0]), float(mn[1]), float(mn[2])], dtype=torch.float32, device=self._device)
            max_t = torch.tensor([float(mx[0]), float(mx[1]), float(mx[2])], dtype=torch.float32, device=self._device)
            if not torch.isfinite(min_t).all() or not torch.isfinite(max_t).all():
                return None
            if torch.any(max_t < min_t):
                return None
            return min_t, max_t
        except Exception:
            return None

    def get_tool_world_aabb(self, *, strict: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mins = []
        maxs = []
        body_name = str(getattr(self.cfg, "task_space_ee_body_name", "Messerkopf"))
        for env_id in range(self.num_envs):
            prim_path = self._find_body_prim_path(body_name, env_id)
            aabb = self._compute_prim_world_aabb(prim_path) if prim_path else None
            if aabb is None:
                if strict:
                    raise RuntimeError(
                        f"Cannot safely track ground trajectory: no geometry AABB for tool body "
                        f"{body_name!r} in env {env_id}. Use --tool_contact_mode manual_offset "
                        "with a validated --tool_tip_local_offset_*."
                    )
                origin = self.get_tool_origin_world_pos()[env_id].detach()
                mins.append(origin)
                maxs.append(origin)
            else:
                mins.append(aabb[0])
                maxs.append(aabb[1])
        return torch.stack(mins, dim=0), torch.stack(maxs, dim=0)

    def get_tool_bottom_world_pos(self) -> torch.Tensor:
        origin = self.get_tool_origin_world_pos()
        mode = str(getattr(self.cfg, "tool_contact_mode", self._tool_contact_mode))
        if mode == "manual_offset":
            return origin + self._manual_tool_offset_world()
        if mode == "ee_origin":
            if not self._tool_contact_warning_printed:
                print(
                    "[HorizonIDFDEnv] WARNING: ee_origin mode does not guarantee tool geometry is above ground.",
                    flush=True,
                )
                self._tool_contact_warning_printed = True
            return origin
        if mode != "geometry_aabb":
            raise ValueError(f"Unsupported tool_contact_mode={mode!r}")
        mn, mx = self.get_tool_world_aabb(strict=True)
        center_xy = 0.5 * (mn[:, :2] + mx[:, :2])
        return torch.cat([center_xy, mn[:, 2:3]], dim=-1)

    def get_tool_bottom_offset_world(self) -> torch.Tensor:
        return self.get_tool_bottom_world_pos() - self.get_tool_origin_world_pos()

    def get_tool_bottom_clearance(self) -> torch.Tensor:
        return self.get_tool_bottom_world_pos()[:, 2] - self._ground_z

    def _body_aabb_min_z(self, body_name: str, env_id: int = 0) -> float | None:
        prim_path = self._find_body_prim_path(body_name, env_id)
        if prim_path is None:
            return None
        aabb = self._compute_prim_world_aabb(prim_path)
        if aabb is None:
            return None
        return float(aabb[0][2].detach().cpu().item())

    def lowest_non_tool_arm_geometry_z(self) -> torch.Tensor:
        body_names = list(getattr(self._uw.robot, "body_names", []))
        tool_name = str(getattr(self.cfg, "task_space_ee_body_name", "Messerkopf"))
        result = []
        for env_id in range(self.num_envs):
            min_z = float("inf")
            for body_name in body_names:
                lname = body_name.lower()
                if body_name == tool_name:
                    continue
                if "wheel" in lname or "reifen" in lname:
                    continue
                if not any(token in body_name for token in ("Ausleger", "Messerkopf", "Drehzapfen", "Cylinder", "Zylinder")):
                    continue
                z = self._body_aabb_min_z(body_name, env_id)
                if z is not None:
                    min_z = min(min_z, z)
            if not math.isfinite(min_z):
                body_pos = self._uw.robot.data.body_link_state_w[env_id, :, :3]
                idx = [
                    i for i, name in enumerate(body_names)
                    if name != tool_name and "wheel" not in name.lower()
                ]
                min_z = float(body_pos[idx, 2].amin().detach().cpu().item()) if idx else float(body_pos[:, 2].amin().detach().cpu().item())
            result.append(min_z)
        return torch.tensor(result, dtype=torch.float32, device=self._device)

    def tool_geometry_report(self) -> dict[str, object]:
        body_name = str(getattr(self.cfg, "task_space_ee_body_name", "Messerkopf"))
        origin = self.get_tool_origin_world_pos()[0].detach().cpu().tolist()
        bottom = self.get_tool_bottom_world_pos()[0].detach().cpu().tolist()
        offset = (self.get_tool_bottom_world_pos()[0] - self.get_tool_origin_world_pos()[0]).detach().cpu().tolist()
        mn, mx = self.get_tool_world_aabb()
        return {
            "ee_body_name": body_name,
            "ee_body_prim_path": self._find_body_prim_path(body_name, 0),
            "ee_origin_world_position": origin,
            "tool_contact_world_position": bottom,
            "tool_bottom_offset_from_ee_origin": offset,
            "tool_contact_mode": str(getattr(self.cfg, "tool_contact_mode", self._tool_contact_mode)),
            "tool_world_aabb_min": mn[0].detach().cpu().tolist(),
            "tool_world_aabb_max": mx[0].detach().cpu().tolist(),
        }

    def _world_to_base(self, pos_w: torch.Tensor) -> torch.Tensor:
        base_pos_w = self._uw.robot.data.root_pos_w
        base_quat_w = self._uw.robot.data.root_quat_w
        base_quat_w = base_quat_w / base_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if pos_w.ndim == 1:
            pos_w = pos_w.unsqueeze(0)
        if pos_w.shape[0] == 1 and self.num_envs > 1:
            pos_w = pos_w.expand(self.num_envs, -1)
        return quat_apply_inverse(base_quat_w, pos_w - base_pos_w)

    def _base_to_world(self, pos_b: torch.Tensor) -> torch.Tensor:
        base_pos_w = self._uw.robot.data.root_pos_w
        base_quat_w = self._uw.robot.data.root_quat_w
        base_quat_w = base_quat_w / base_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if pos_b.ndim == 1:
            pos_b = pos_b.unsqueeze(0)
        if pos_b.shape[0] == 1 and self.num_envs > 1:
            pos_b = pos_b.expand(self.num_envs, -1)
        return base_pos_w + quat_apply(base_quat_w, pos_b)

    def discover_controlled_frame(self) -> dict[str, object]:
        geom = self.tool_geometry_report()
        ee_world = self.get_tool_origin_world_pos()[0].detach().cpu().tolist()
        tool_world = self.get_tool_bottom_world_pos()[0].detach().cpu().tolist()
        ee_base = self._tcp_pos_base_frame()[0].detach().cpu().tolist()
        root_q = self._uw.robot.data.root_quat_w[0].detach().cpu().tolist()
        root_p = self._uw.robot.data.root_pos_w[0].detach().cpu().tolist()
        if self._is_task_space:
            if self.num_actions == 2:
                controlled_axes = "base_frame_YZ"
                can_ground_xy = False
            elif self.num_actions >= 3:
                controlled_axes = "base_frame_XYZ"
                can_ground_xy = True
            else:
                controlled_axes = f"base_frame_first_{self.num_actions}_axes"
                can_ground_xy = False
        else:
            controlled_axes = "cylinder_space"
            can_ground_xy = False
        return {
            "world_axes": "world X/Y horizontal, world Z vertical; ground plane is world XY at fixed world Z",
            "base_frame_position_world": root_p,
            "base_frame_quat_world_wxyz": root_q,
            "current_ee_origin_world_position": ee_world,
            "current_tool_contact_world_position": tool_world,
            "current_ee_base_frame_position": ee_base,
            "ee_body_name": self.cfg.task_space_ee_body_name,
            "ee_body_prim_path": geom["ee_body_prim_path"],
            "tool_bottom_offset_from_ee_origin": geom["tool_bottom_offset_from_ee_origin"],
            "controlled_frame": "base_frame" if self._is_task_space else "cylinder_space",
            "controlled_axes": controlled_axes,
            "can_represent_true_ground_xy_path": can_ground_xy,
            "tracking_frame_note": (
                "Ground trajectories track the tool/contact bottom point, converted to an EE-origin command."
                if str(getattr(self.cfg, "tool_contact_mode", self._tool_contact_mode)) != "ee_origin"
                else "WARNING: ee_origin mode tracks the EE origin and does not guarantee tool geometry clearance."
            ),
        }

    def _expand_task_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] == 3:
            return action
        if action.shape[-1] == 2:
            expanded = torch.zeros((action.shape[0], 3), dtype=action.dtype, device=action.device)
            expanded[:, 1:3] = action
            return expanded
        if action.shape[-1] > 3:
            return action[:, :3]
        expanded = torch.zeros((action.shape[0], 3), dtype=action.dtype, device=action.device)
        expanded[:, : action.shape[-1]] = action
        return expanded

    def _coerce_action_dim(self, action: torch.Tensor) -> torch.Tensor:
        """Pad or trim caller-provided actions to the configured action space."""
        if action.shape[-1] == self.num_actions:
            return action
        fixed = torch.zeros((action.shape[0], self.num_actions), dtype=action.dtype, device=action.device)
        n = min(action.shape[-1], self.num_actions)
        fixed[:, :n] = action[:, :n]
        return fixed

    def set_ground_target_world(self, target_world: torch.Tensor) -> None:
        """Set target tool/contact-bottom world point and command the required EE origin."""
        target_world = target_world.to(device=self._device, dtype=torch.float32)
        if target_world.ndim == 1:
            target_world = target_world.unsqueeze(0)
        if target_world.shape[0] == 1 and self.num_envs > 1:
            target_world = target_world.expand(self.num_envs, -1)
        target_world = target_world.clone()
        min_z = self._ground_z + self._tool_ground_clearance
        target_world[:, 2] = torch.clamp(target_world[:, 2], min=min_z)
        self._ground_target_tool_bottom_world = target_world.detach().clone()
        offset = self.get_tool_bottom_offset_world().detach()
        ee_origin_target_world = target_world - offset
        target_base = self._world_to_base(ee_origin_target_world)
        self._uw._command_pos_w[:, :3] = target_base[:, :3]
        self._uw._command_quat_w[:, 0] = 1.0
        self._uw._command_quat_w[:, 1:] = 0.0

    def current_ground_target_world(self) -> torch.Tensor:
        return self._ground_target_tool_bottom_world.detach().clone()

    def current_ee_origin_target_world(self) -> torch.Tensor:
        return self._base_to_world(self._uw._command_pos_w[:, :3])

    def configure_ground_trajectory(
        self,
        mode: str | None = None,
        workspace_json: str | None = None,
        trajectory_plane: str | None = None,
        trajectory_scale: float | None = None,
        visualize_target_trajectory: bool | None = None,
        visualize_actual_ee_trail: bool | None = None,
        num_points: int | None = None,
    ) -> torch.Tensor:
        self._trajectory_mode = str(mode or self._trajectory_mode)
        self._trajectory_plane = str(trajectory_plane or self._trajectory_plane)
        self._ground_workspace_json = str(workspace_json or self._ground_workspace_json)
        if trajectory_scale is not None:
            self._trajectory_scale = float(trajectory_scale)
        points = int(num_points or getattr(self.cfg, "ground_trajectory_points", 200))
        workspace = self._load_ground_workspace(self._ground_workspace_json)
        path_np, params, limitation = self._generate_ground_path_np(
            self._trajectory_mode,
            workspace,
            self._trajectory_plane,
            self._trajectory_scale,
            points,
        )
        self._ground_path_world = torch.as_tensor(path_np, dtype=torch.float32, device=self._device)
        self._ground_path_index = 0
        self._ground_path_completion.zero_()
        self._ground_limitation = limitation
        self._ground_path_params = params
        self.set_ground_target_world(self._ground_path_world[0])
        vis_target = bool(
            getattr(self.cfg, "visualize_target_trajectory", False)
            if visualize_target_trajectory is None
            else visualize_target_trajectory
        )
        vis_trail = bool(
            getattr(self.cfg, "visualize_actual_ee_trail", False)
            if visualize_actual_ee_trail is None
            else visualize_actual_ee_trail
        )
        if vis_target or vis_trail:
            self._setup_ground_visualization(vis_target, vis_trail)
            self._update_ground_visualization()
        return self._ground_path_world

    def _load_ground_workspace(self, workspace_json: str) -> dict:
        path = Path(workspace_json)
        if path.is_file():
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        ee_world = self.get_tool_bottom_world_pos()[0].detach().cpu().numpy()
        span = 0.8
        return {
            "ground_z": self._ground_z,
            "tool_ground_clearance": self._tool_ground_clearance,
            "target_world_z": self._ground_z + self._tool_ground_clearance,
            "safe_x_min": float(ee_world[0] - span),
            "safe_x_max": float(ee_world[0] + span),
            "safe_y_min": float(ee_world[1] - span),
            "safe_y_max": float(ee_world[1] + span),
            "largest_line_endpoints": [
                [float(ee_world[0] - 0.6 * span), float(ee_world[1]), self._ground_z + self._tool_ground_clearance],
                [float(ee_world[0] + 0.6 * span), float(ee_world[1]), self._ground_z + self._tool_ground_clearance],
            ],
            "semicircle_center": [float(ee_world[0]), float(ee_world[1]), self._ground_z + self._tool_ground_clearance],
            "semicircle_radius": 0.5 * span,
            "mower_sweep_bounds": {
                "x_min": float(ee_world[0] - 0.6 * span),
                "x_max": float(ee_world[0] + 0.6 * span),
                "y_min": float(ee_world[1] - 0.4 * span),
                "y_max": float(ee_world[1] + 0.4 * span),
            },
            "mower_sweep_amplitude": 0.25 * span,
            "mower_sweep_length": 1.2 * span,
        }

    def _generate_ground_path_np(
        self,
        mode: str,
        workspace: dict,
        trajectory_plane: str,
        scale: float,
        points: int,
    ) -> tuple[np.ndarray, dict, str]:
        z = self._ground_z + self._tool_ground_clearance
        x_min = float(workspace.get("safe_x_min", -1.0))
        x_max = float(workspace.get("safe_x_max", 1.0))
        y_min = float(workspace.get("safe_y_min", -1.0))
        y_max = float(workspace.get("safe_y_max", 1.0))
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        center = workspace.get("semicircle_center", [cx, cy, z])
        cx = float(center[0])
        cy = float(center[1])
        radius_max = min(cx - x_min, x_max - cx, cy - y_min, y_max - cy)
        radius = float(workspace.get("semicircle_radius", max(radius_max, 0.05))) * float(scale)
        radius = max(0.02, min(radius, max(radius_max, 0.02)))
        limitation = ""
        true_ground_xy = self._is_task_space and self.num_actions >= 3 and trajectory_plane == "ground_xy"
        reduced = not true_ground_xy
        if reduced:
            limitation = (
                "Current task interface cannot command world X/Y independently; "
                "using fixed world height and varying the available horizontal coordinate only."
            )

        n = max(int(points), 2)
        if mode == "ground_semicircle":
            theta = np.linspace(0.0, math.pi, n)
            if reduced:
                x = np.full_like(theta, cx)
                y = cy + radius * np.cos(theta)
            else:
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
        elif mode == "ground_straight_line":
            endpoints = workspace.get("largest_line_endpoints")
            if endpoints and len(endpoints) >= 2:
                p0 = np.asarray(endpoints[0], dtype=np.float64)
                p1 = np.asarray(endpoints[1], dtype=np.float64)
            else:
                p0 = np.asarray([x_min, cy, z], dtype=np.float64)
                p1 = np.asarray([x_max, cy, z], dtype=np.float64)
            mid = 0.5 * (p0 + p1)
            p0 = mid + 0.9 * float(scale) * (p0 - mid)
            p1 = mid + 0.9 * float(scale) * (p1 - mid)
            if reduced:
                p0[0] = mid[0]
                p1[0] = mid[0]
            s = np.linspace(0.0, 1.0, n)
            x = p0[0] * (1.0 - s) + p1[0] * s
            y = p0[1] * (1.0 - s) + p1[1] * s
        elif mode == "ground_mower_sweep":
            bounds = workspace.get("mower_sweep_bounds", {})
            bx0 = float(bounds.get("x_min", x_min))
            bx1 = float(bounds.get("x_max", x_max))
            by0 = float(bounds.get("y_min", y_min))
            by1 = float(bounds.get("y_max", y_max))
            mx = 0.5 * (bx0 + bx1)
            my = 0.5 * (by0 + by1)
            half_y = max(0.02, 0.5 * (by1 - by0))
            amp = min(float(workspace.get("mower_sweep_amplitude", 0.7 * half_y)), 0.7 * half_y) * float(scale)
            waves = 3 if abs(bx1 - bx0) > 1.0 else 2
            s = np.linspace(0.0, 1.0, n)
            if reduced:
                x = np.full_like(s, mx)
                y = my + amp * np.sin(2.0 * math.pi * waves * s)
            else:
                x = bx0 + s * (bx1 - bx0)
                y = my + amp * np.sin(2.0 * math.pi * waves * s)
        else:
            raise ValueError(f"Unsupported trajectory_mode={mode!r}")

        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)
        z_arr = np.full_like(x, z)
        path = np.stack([x, y, z_arr], axis=-1).astype(np.float32)
        params = {
            "mode": mode,
            "trajectory_plane": trajectory_plane,
            "reduced_to_single_horizontal_axis": bool(reduced),
            "target_world_z": float(z),
            "tracking_point": "tool_bottom_contact",
            "semicircle_radius": float(radius),
            "x_min": float(np.min(x)),
            "x_max": float(np.max(x)),
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y)),
            "points": int(n),
        }
        return path, params, limitation

    def compute_ground_tracking_action(self, gain: float = 1.0) -> torch.Tensor:
        target_base = self._uw._command_pos_w[:, :3]
        ee_base = self._tcp_pos_base_frame()
        err = target_base - ee_base
        desired = torch.clamp(err * float(gain) / max(self._ee_action_scale, 1.0e-6), -1.0, 1.0)
        if self.num_actions == 2:
            return desired[:, 1:3]
        if self.num_actions >= 3:
            return desired[:, : self.num_actions]
        return desired[:, : self.num_actions]

    def advance_ground_trajectory(self, steps: int = 1) -> None:
        if self._ground_path_world is None or self._ground_path_world.numel() == 0:
            return
        last = self._ground_path_world.shape[0] - 1
        self._ground_path_index = min(last, self._ground_path_index + int(steps))
        self._ground_path_completion[:] = float(self._ground_path_index) / max(float(last), 1.0)
        self.set_ground_target_world(self._ground_path_world[self._ground_path_index])

    def ground_safety_metrics(self, target_world: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if target_world is None:
            target_world = self.current_ground_target_world()
        if target_world.ndim == 1:
            target_world = target_world.unsqueeze(0)
        ee_origin = self.get_tool_origin_world_pos()
        tool_bottom = self.get_tool_bottom_world_pos()
        body_pos = self._uw.robot.data.body_link_state_w[:, :, :3]
        arm_min_z = self.lowest_non_tool_arm_geometry_z()
        root_pos = self._uw.robot.data.root_pos_w
        all_finite = (
            torch.isfinite(target_world).all(dim=-1)
            & torch.isfinite(ee_origin).all(dim=-1)
            & torch.isfinite(tool_bottom).all(dim=-1)
            & torch.isfinite(body_pos).all(dim=(1, 2))
            & torch.isfinite(root_pos).all(dim=-1)
        )
        max_abs_body = body_pos.abs().amax(dim=(1, 2))
        root_motion = torch.linalg.norm(root_pos - self._root_hold_pose[:, :3], dim=-1)
        lo, hi = self._joint_limits()
        cyl = self._uw.robot.data.joint_pos[:, self._uw._cylinder_joint_ids]
        span = (hi - lo).clamp(min=1.0e-6)
        near_limit = ((cyl - lo) < 0.02 * span) | ((hi - cyl) < 0.02 * span)
        tracking_error = torch.linalg.norm(tool_bottom - target_world[:, :3], dim=-1)
        return {
            "ground_z": torch.full((self.num_envs,), float(self._ground_z), device=self._device),
            "ee_origin_world_z": ee_origin[:, 2],
            "tool_bottom_world_z": tool_bottom[:, 2],
            "tool_bottom_clearance": tool_bottom[:, 2] - self._ground_z,
            "target_tool_bottom_world_z": target_world[:, 2],
            "target_tool_bottom_clearance": target_world[:, 2] - self._ground_z,
            "lowest_arm_geometry_world_z": arm_min_z,
            "arm_geometry_clearance": arm_min_z - self._ground_z,
            "target_ground_clearance": target_world[:, 2] - self._ground_z,
            "ee_ground_clearance": tool_bottom[:, 2] - self._ground_z,
            "arm_link_ground_clearance": arm_min_z - self._ground_z,
            "max_abs_body_world_position": max_abs_body,
            "root_motion_norm": root_motion,
            "nan_inf_count": (~all_finite).long(),
            "cylinder_limit_fraction": near_limit.float().mean(dim=-1),
            "tracking_error": tracking_error,
            "path_completion_fraction": self._ground_path_completion.detach().clone(),
        }

    def _setup_ground_visualization(self, target_path: bool, trail: bool) -> None:
        try:
            import isaaclab.sim as sim_utils
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()

            def spawn_once(prim_path: str, cfg, pos: tuple[float, float, float]) -> None:
                prim = stage.GetPrimAtPath(prim_path)
                if prim and prim.IsValid():
                    self._set_prim_translation(prim_path, pos)
                else:
                    cfg.func(prim_path, cfg, translation=pos)

            self._ground_path_marker_paths = []
            marker_cfg = sim_utils.MeshCuboidCfg(
                size=(0.045, 0.045, 0.045),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.2)),
            )
            if target_path and self._ground_path_world is not None:
                path = self._ground_path_world.detach().cpu().numpy()
                stride = max(1, int(math.ceil(len(path) / 128)))
                for idx, pos in enumerate(path[::stride]):
                    prim_path = f"/World/ground_trajectory/path_{idx:03d}"
                    spawn_once(prim_path, marker_cfg, tuple(float(v) for v in pos))
                    self._ground_path_marker_paths.append(prim_path)
            current_cfg = sim_utils.MeshCuboidCfg(
                size=(0.12, 0.12, 0.12),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
            )
            spawn_once(self._ground_current_marker_path, current_cfg, (0.0, 0.0, 0.0))
            ee_origin_cfg = sim_utils.MeshCuboidCfg(
                size=(0.10, 0.10, 0.10),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.0)),
            )
            tool_contact_cfg = sim_utils.MeshCuboidCfg(
                size=(0.08, 0.08, 0.08),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.95, 1.0)),
            )
            target_contact_cfg = sim_utils.MeshCuboidCfg(
                size=(0.11, 0.11, 0.11),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.75)),
            )
            spawn_once(self._ground_ee_origin_marker_path, ee_origin_cfg, (0.0, 0.0, 0.0))
            if bool(getattr(self.cfg, "debug_draw_tool_contact_point", False)):
                spawn_once(self._ground_tool_contact_marker_path, tool_contact_cfg, (0.0, 0.0, 0.0))
                spawn_once(self._ground_target_contact_marker_path, target_contact_cfg, (0.0, 0.0, 0.0))
            if bool(getattr(self.cfg, "debug_draw_tool_aabb", False)):
                mn, mx = self.get_tool_world_aabb()
                aabb_center = (0.5 * (mn[0] + mx[0])).detach().cpu().tolist()
                aabb_size = torch.clamp(mx[0] - mn[0], min=0.01).detach().cpu().tolist()
                aabb_cfg = sim_utils.MeshCuboidCfg(
                    size=tuple(float(v) for v in aabb_size),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), opacity=0.25),
                )
                spawn_once(self._ground_tool_aabb_marker_path, aabb_cfg, tuple(float(v) for v in aabb_center))
            self._ground_trail_marker_paths = []
            if trail:
                trail_cfg = sim_utils.MeshCuboidCfg(
                    size=(0.035, 0.035, 0.035),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.55, 1.0)),
                )
                trail_len = int(getattr(self.cfg, "ground_actual_trail_length", 100))
                for idx in range(max(0, trail_len)):
                    prim_path = f"/World/ground_trajectory/trail_{idx:03d}"
                    spawn_once(prim_path, trail_cfg, (0.0, 0.0, -1000.0))
                    self._ground_trail_marker_paths.append(prim_path)
            _ = UsdGeom
            self._ground_visuals_ready = True
        except Exception as exc:
            if not self._ground_visual_warning_printed:
                print(f"[HorizonIDFDEnv] WARNING: ground visualization unavailable: {exc}", flush=True)
                self._ground_visual_warning_printed = True
            self._ground_visuals_ready = False

    def _set_prim_translation(self, prim_path: str, pos: tuple[float, float, float]) -> None:
        from pxr import UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(pos)

    def _update_ground_visualization(self) -> None:
        if not self._ground_visuals_ready:
            return
        try:
            target = self.current_ground_target_world()[0].detach().cpu().tolist()
            self._set_prim_translation(self._ground_current_marker_path, tuple(float(v) for v in target))
            ee_origin = self.get_tool_origin_world_pos()[0].detach().cpu().tolist()
            tool_bottom = self.get_tool_bottom_world_pos()[0].detach().cpu().tolist()
            self._set_prim_translation(self._ground_ee_origin_marker_path, tuple(float(v) for v in ee_origin))
            if bool(getattr(self.cfg, "debug_draw_tool_contact_point", False)):
                self._set_prim_translation(self._ground_tool_contact_marker_path, tuple(float(v) for v in tool_bottom))
                self._set_prim_translation(self._ground_target_contact_marker_path, tuple(float(v) for v in target))
            if bool(getattr(self.cfg, "debug_draw_tool_aabb", False)):
                mn, mx = self.get_tool_world_aabb()
                center = (0.5 * (mn[0] + mx[0])).detach().cpu().tolist()
                self._set_prim_translation(self._ground_tool_aabb_marker_path, tuple(float(v) for v in center))
            if self._ground_trail_marker_paths:
                self._ground_trail_points.append(tuple(float(v) for v in tool_bottom))
                max_len = len(self._ground_trail_marker_paths)
                self._ground_trail_points = self._ground_trail_points[-max_len:]
                hidden = (0.0, 0.0, -1000.0)
                for idx, prim_path in enumerate(self._ground_trail_marker_paths):
                    if idx < len(self._ground_trail_points):
                        self._set_prim_translation(prim_path, self._ground_trail_points[idx])
                    else:
                        self._set_prim_translation(prim_path, hidden)
        except Exception as exc:
            if not self._ground_visual_warning_printed:
                print(f"[HorizonIDFDEnv] WARNING: ground visualization update failed: {exc}", flush=True)
                self._ground_visual_warning_printed = True
            self._ground_visuals_ready = False


    def _is_stage0(self) -> bool:
        return str(getattr(self.cfg, "ppo_stage", "full_mowing")) == "local_reach_stage0"

    def _resample_stage0_target(self, env_ids=None) -> None:
        """Stage 0: sample target in small local box around current TCP pose."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        if env_ids.numel() == 0:
            return
        n = len(env_ids)
        level = int(getattr(self.cfg, "stage0_level", 0))
        if level == 0:
            dx = getattr(self.cfg, "stage0_l0_target_box_dx", (-0.05, 0.05))
            dy = getattr(self.cfg, "stage0_l0_target_box_dy", (-0.05, 0.05))
            dz = getattr(self.cfg, "stage0_l0_target_box_dz", (-0.02, 0.02))
        elif level == 1:
            dx = getattr(self.cfg, "stage0_l1_target_box_dx", (-0.10, 0.10))
            dy = getattr(self.cfg, "stage0_l1_target_box_dy", (-0.10, 0.10))
            dz = getattr(self.cfg, "stage0_l1_target_box_dz", (-0.03, 0.03))
        else:
            dx = getattr(self.cfg, "stage0_l2_target_box_dx", (-0.15, 0.15))
            dy = getattr(self.cfg, "stage0_l2_target_box_dy", (-0.15, 0.15))
            dz = getattr(self.cfg, "stage0_l2_target_box_dz", (-0.05, 0.05))
        tcp = self._tcp_pos_base_frame()[env_ids].detach().clone()
        rand_x = (torch.rand(n, device=self._device) * (dx[1] - dx[0]) + dx[0])
        rand_y = (torch.rand(n, device=self._device) * (dy[1] - dy[0]) + dy[0])
        rand_z = (torch.rand(n, device=self._device) * (dz[1] - dz[0]) + dz[0])
        target = tcp + torch.stack([rand_x, rand_y, rand_z], dim=-1)
        # Clamp into safe workspace and outside cab
        wx = getattr(self.cfg, "horizon_workspace_x_range", (-0.5, 7.0))
        wy = getattr(self.cfg, "horizon_workspace_y_range", (-3.6, 3.0))
        wz = getattr(self.cfg, "horizon_workspace_z_range", (-0.4, 3.0))
        cx = getattr(self.cfg, "horizon_cab_x_range", (-2.5, 1.0))
        cy = getattr(self.cfg, "horizon_cab_y_range", (-1.35, 1.35))
        cz = getattr(self.cfg, "horizon_cab_z_range", (-0.1, 2.4))
        target[:, 0] = torch.clamp(target[:, 0], float(wx[0]) + 0.05, float(wx[1]) - 0.05)
        target[:, 1] = torch.clamp(target[:, 1], float(wy[0]) + 0.05, float(wy[1]) - 0.05)
        target[:, 2] = torch.clamp(target[:, 2], float(wz[0]) + 0.05, float(wz[1]) - 0.05)
        # Push out of cab region (full 3D check)
        in_cab_3d = (
            (target[:, 0] >= cx[0]) & (target[:, 0] <= cx[1])
            & (target[:, 1] >= cy[0]) & (target[:, 1] <= cy[1])
            & (target[:, 2] >= cz[0]) & (target[:, 2] <= cz[1])
        )
        target[in_cab_3d, 0] = float(cx[1]) + 0.05
        self._uw._command_pos_w[env_ids, 0] = target[:, 0]
        self._uw._command_pos_w[env_ids, 1] = target[:, 1]
        self._uw._command_pos_w[env_ids, 2] = target[:, 2]
        self._uw._command_quat_w[env_ids, 0] = 1.0
        self._uw._command_quat_w[env_ids, 1:] = 0.0
    def _workspace_guard_rows(self) -> torch.Tensor:
        outside_workspace, inside_cab = self._workspace_guard_components()
        if not bool(getattr(self.cfg, "enable_horizon_workspace_guard", True)):
            outside_workspace = torch.zeros_like(outside_workspace)
        if not bool(getattr(self.cfg, "enable_horizon_cab_guard", True)):
            inside_cab = torch.zeros_like(inside_cab)
        return outside_workspace | inside_cab

    def _workspace_guard_components(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(getattr(self.cfg, "enable_horizon_workspace_guard", True)):
            outside_workspace = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        else:
            tcp = self._tcp_pos_base_frame()
            wx = getattr(self.cfg, "horizon_workspace_x_range", (1.0, 7.0))
            wy = getattr(self.cfg, "horizon_workspace_y_range", (-3.0, 3.0))
            wz = getattr(self.cfg, "horizon_workspace_z_range", (-0.4, 3.0))
            outside_workspace = (
                (tcp[:, 0] < float(wx[0]))
                | (tcp[:, 0] > float(wx[1]))
                | (tcp[:, 1] < float(wy[0]))
                | (tcp[:, 1] > float(wy[1]))
                | (tcp[:, 2] < float(wz[0]))
                | (tcp[:, 2] > float(wz[1]))
            )
        tcp = self._tcp_pos_base_frame()
        if not bool(getattr(self.cfg, "enable_horizon_cab_guard", True)):
            return outside_workspace, torch.zeros_like(outside_workspace)
        cx = getattr(self.cfg, "horizon_cab_x_range", (-2.5, 1.0))
        cy = getattr(self.cfg, "horizon_cab_y_range", (-1.35, 1.35))
        cz = getattr(self.cfg, "horizon_cab_z_range", (-0.1, 2.4))
        inside_cab = (
            (tcp[:, 0] >= float(cx[0]))
            & (tcp[:, 0] <= float(cx[1]))
            & (tcp[:, 1] >= float(cy[0]))
            & (tcp[:, 1] <= float(cy[1]))
            & (tcp[:, 2] >= float(cz[0]))
            & (tcp[:, 2] <= float(cz[1]))
        )
        return outside_workspace, inside_cab

    def _cab_clearance(self) -> torch.Tensor:
        tcp = self._tcp_pos_base_frame()
        cx = getattr(self.cfg, "horizon_cab_x_range", (-2.5, 1.0))
        cy = getattr(self.cfg, "horizon_cab_y_range", (-1.35, 1.35))
        cz = getattr(self.cfg, "horizon_cab_z_range", (-0.1, 2.4))
        lo = torch.tensor([float(cx[0]), float(cy[0]), float(cz[0])], device=self._device)
        hi = torch.tensor([float(cx[1]), float(cy[1]), float(cz[1])], device=self._device)
        below = torch.clamp(lo - tcp, min=0.0)
        above = torch.clamp(tcp - hi, min=0.0)
        outside_distance = torch.linalg.norm(below + above, dim=-1)
        inside = ((tcp >= lo) & (tcp <= hi)).all(dim=-1)
        face_distance = torch.minimum(tcp - lo, hi - tcp).min(dim=-1).values
        return torch.where(inside, -face_distance, outside_distance)

    def _guard_target(self, raw_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = self._joint_limits()
        clamped_target = torch.clamp(raw_target, lo, hi)
        guard_flags = (clamped_target - raw_target).abs() > 1.0e-7
        guarded_rows = self._workspace_guard_rows()
        if guarded_rows.any():
            clamped_target = torch.where(
                guarded_rows.unsqueeze(-1),
                self._cylinder_hold_target,
                clamped_target,
            )
            guard_flags = guard_flags | guarded_rows.unsqueeze(-1)
        clamped_target = torch.nan_to_num(clamped_target, nan=0.0, posinf=0.0, neginf=0.0)
        guard_flags = guard_flags.float()
        self._last_guard_fraction = guard_flags.mean(dim=-1).detach().clone()
        return clamped_target, guard_flags

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

    def _smooth_ramp(
        self,
        delta_cmd_H: torch.Tensor,
        cq0: torch.Tensor,
        neutral_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute smooth position ramp over H steps and keep final hold target live."""
        dx_isaac = self._uw._cylinder_adapter.simscape_delta_to_isaac_delta(delta_cmd_H)
        raw_final_target = cq0 + dx_isaac
        raw_final_target = torch.where(
            neutral_action.unsqueeze(-1),
            self._cylinder_hold_target,
            raw_final_target,
        )
        final_target, guard_flags = self._guard_target(raw_final_target)

        zero_action = torch.zeros((self.num_envs, 5), device=self._device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        for i in range(1, self._H + 1):
            frac = i / self._H
            target = cq0 + frac * (final_target - cq0)
            self._uw.set_debug_manual_cylinder_targets(
                target.detach().clone(),
                tool_position_target=self._tool_hold_target,
            )
            self._pin_root_pose_if_enabled()
            _, _, term_i, trunc_i, _ = self._env.step(zero_action)
            self._pin_root_pose_if_enabled()
            terminated |= term_i
            truncated |= trunc_i

        self._cylinder_hold_target = final_target.detach().clone()
        self._uw.set_debug_manual_cylinder_targets(
            self._cylinder_hold_target,
            tool_position_target=self._tool_hold_target,
        )
        return final_target, guard_flags, terminated, truncated

    def _build_observation(self) -> torch.Tensor:
        if self._is_task_space:
            return build_high_level_observation(self._uw, self._expand_task_action(self._prev_hl_action))
        else:
            return build_horizon_observation(
                self._uw, self._prev_action,
                self._prev_delta_x_des, self._prev_fd_delta,
                self._prev_cmd_delta, self._prev_meas_delta,
                self._prev_guard_flags,
            )

    def reset(self):
        self._env.reset()
        self._prev_action.zero_()
        self._prev_delta_x_des.zero_()
        self._prev_fd_delta.zero_()
        self._prev_meas_delta.zero_()
        self._prev_cmd_delta.zero_()
        self._prev_guard_flags.zero_()
        self._last_guard_fraction.zero_()
        self._last_reward_terms = {}
        self._success_hold_steps.zero_()
        self._reset_episode_buffers(torch.arange(self.num_envs, device=self._device))
        self._refresh_root_hold_from_current()
        self._refresh_hold_target_from_current()
        self._settle_reset_hold()
        # -- Measure J_eff through full deployment chain (delta_x_des_H -> IDFD -> smooth ramp -> EE) --
        if self._is_task_space and self._ee_controller is not None:
            # Verify base is static
            _base_vel = torch.norm(self._uw.robot.data.root_lin_vel_w, dim=-1)
            _base_ang = torch.norm(self._uw.robot.data.root_ang_vel_w, dim=-1)
            _base_static = (_base_vel < 0.01) & (_base_ang < 0.01)
            if not _base_static.all():
                print(f"[HorizonIDFDEnv] WARNING: base not static before J_eff: lin={_base_vel.max().item():.4f} ang={_base_ang.max().item():.4f}", flush=True)
            # Save full robot state
            _saved_root = self._uw.robot.data.root_state_w.clone()
            _saved_joint_pos = self._uw.robot.data.joint_pos.clone()
            _saved_joint_vel = self._uw.robot.data.joint_vel.clone()
            _saved_hold = self._cylinder_hold_target.clone()
            _saved_tool = self._tool_hold_target.clone()
            _saved_prev = self._prev_action.clone()
            _saved_pd = self._prev_delta_x_des.clone()
            _saved_pf = self._prev_fd_delta.clone()
            _saved_pm = self._prev_meas_delta.clone()
            _saved_pc = self._prev_cmd_delta.clone()
            _saved_pg = self._prev_guard_flags.clone()

            def _restore_full():
                self._uw.robot.write_root_state_to_sim(_saved_root)
                self._uw.robot.write_joint_state_to_sim(
                    _saved_joint_pos, _saved_joint_vel, None,
                    torch.arange(self.num_envs, device=self._device))
                self._cylinder_hold_target = _saved_hold.clone()
                self._tool_hold_target = _saved_tool.clone()
                self._prev_action = _saved_prev.clone()
                self._prev_delta_x_des = _saved_pd.clone()
                self._prev_fd_delta = _saved_pf.clone()
                self._prev_meas_delta = _saved_pm.clone()
                self._prev_cmd_delta = _saved_pc.clone()
                self._prev_guard_flags = _saved_pg.clone()
                self._uw.set_debug_manual_cylinder_targets(
                    _saved_hold, tool_position_target=_saved_tool)
                self._pin_root_pose_if_enabled()
                self._env.step(torch.zeros((self.num_envs, 5), device=self._device))
                self._pin_root_pose_if_enabled()

            def _deploy(dx_des_H):
                """Apply delta_x_des_H through ID/FD and the smooth PhysX deployment."""
                cq0 = self._uw.robot.data.joint_pos[:, self._uw._cylinder_joint_ids].detach().clone()
                x0, xd0 = self._uw._cylinder_adapter.current_from_cylinder_joints(
                    cq0,
                    self._uw.robot.data.joint_vel[:, self._uw._cylinder_joint_ids],
                )
                delta_cmd = self._idfd_rollout(dx_des_H, x0, xd0)
                final_target, _, _, _ = self._smooth_ramp(
                    delta_cmd, cq0,
                    torch.zeros(self.num_envs, dtype=torch.bool, device=self._device))
                self._cylinder_hold_target = final_target.detach().clone()
                ee = self._tcp_pos_base_frame()
                return ee, _restore_full

            J, cond, S = self._ee_controller.measure_effective_jacobian(_deploy, self.num_envs)
            self._jacobian_valid[:] = torch.isfinite(J).all(dim=(1,2)) & (
                cond < 5000.0) & _base_static
            if self._jacobian_valid.all():
                print(f"[HorizonIDFDEnv] J_eff measured: rank={self._ee_controller.effective_rank()} "
                      f"cond={cond.mean().item():.1f} base_static={_base_static.all().item()}", flush=True)
            else:
                print(f"[HorizonIDFDEnv] WARNING: J_eff invalid: finite={torch.isfinite(J).all().item()} "
                      f"cond={cond.mean().item():.1f} base_static={_base_static.all().item()}", flush=True)
        # -- Stage-0: resample local target after reset --
        if self._is_stage0():
            self._resample_stage0_target()
        if self._trajectory_mode.startswith("ground_"):
            self._ground_tracking_clearance = self._tool_ground_clearance
            self.configure_ground_trajectory(
                mode=self._trajectory_mode,
                workspace_json=self._ground_workspace_json,
                trajectory_plane=self._trajectory_plane,
                trajectory_scale=self._trajectory_scale,
            )
        return self._build_observation(), {}

    def _reset_episode_buffers(self, env_ids: torch.Tensor) -> None:
        self._episode_return[env_ids] = 0.0
        self._horizon_episode_length[env_ids] = 0
        self._episode_ee_error_start[env_ids] = compute_tracking_error(self._uw).detach()[env_ids]
        self._episode_guard_sum[env_ids] = 0.0
        self._episode_clip_sum[env_ids] = 0.0
        self._episode_hydraulic_sum[env_ids] = 0.0
        self._episode_action_norm_sum[env_ids] = 0.0
        self._episode_action_rate_sum[env_ids] = 0.0
        self._episode_delta_x_des_sum[env_ids] = 0.0
        self._episode_delta_x_meas_sum[env_ids] = 0.0
        self._episode_min_cab_clearance[env_ids] = float("inf")
        self._success_hold_steps[env_ids] = 0
        self._ws_consecutive_steps[env_ids] = 0

    def _base_done_masks(self) -> dict[str, torch.Tensor]:
        zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        raw = getattr(self._uw, "_last_done_cause_masks", {})
        return {name: raw.get(name, zero).detach().clone() for name in self.DONE_CAUSES}

    def _update_episode_sums(
        self,
        reward: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        delta_x_des_H: torch.Tensor,
        delta_x_meas_H: torch.Tensor,
        guard_fraction: torch.Tensor,
        clipping_fraction: torch.Tensor,
        hydraulic_error: torch.Tensor,
    ) -> None:
        self._episode_return += reward.detach()
        self._horizon_episode_length += 1
        self._episode_guard_sum += guard_fraction.detach()
        self._episode_clip_sum += clipping_fraction.detach()
        self._episode_hydraulic_sum += hydraulic_error.detach()
        self._episode_action_norm_sum += torch.linalg.norm(action.detach(), dim=-1)
        self._episode_action_rate_sum += torch.linalg.norm((action - prev_action).detach(), dim=-1)
        self._episode_delta_x_des_sum += delta_x_des_H.detach().abs().mean(dim=-1)
        self._episode_delta_x_meas_sum += delta_x_meas_H.detach().abs().mean(dim=-1)
        self._episode_min_cab_clearance = torch.minimum(self._episode_min_cab_clearance, self._cab_clearance().detach())

    def _episode_mean(self, values: torch.Tensor, env_id: int) -> float:
        length = max(int(self._horizon_episode_length[env_id].detach().cpu().item()), 1)
        return float((values[env_id] / length).detach().cpu().item())

    def _record_done_episodes(
        self,
        done: torch.Tensor,
        cause_masks: dict[str, torch.Tensor],
        final_reward: torch.Tensor,
        final_ee_error: torch.Tensor,
    ) -> None:
        env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        rows: list[dict[str, float | int | str]] = []
        for env_tensor in env_ids:
            env_id = int(env_tensor.detach().cpu().item())
            matched = [name for name in self.DONE_CAUSES if bool(cause_masks[name][env_id].detach().cpu().item())]
            if not matched:
                matched = ["unknown_done"]
            length = int(self._horizon_episode_length[env_id].detach().cpu().item())
            start_err = float(self._episode_ee_error_start[env_id].detach().cpu().item())
            end_err = float(final_ee_error[env_id].detach().cpu().item())
            row = {
                "episode_index": len(self._episode_records) + len(rows),
                "env_id": env_id,
                "cause": "|".join(matched),
                "episode_length": length,
                "final_reward": float(final_reward[env_id].detach().cpu().item()),
                "episode_return": float(self._episode_return[env_id].detach().cpu().item()),
                "final_ee_error": end_err,
                "ee_error_before_episode": start_err,
                "ee_error_improvement": start_err - end_err,
                "min_cab_clearance": float(self._episode_min_cab_clearance[env_id].detach().cpu().item()),
                "guard_fraction": self._episode_mean(self._episode_guard_sum, env_id),
                "clipping_fraction": self._episode_mean(self._episode_clip_sum, env_id),
                "hydraulic_consistency_error": self._episode_mean(self._episode_hydraulic_sum, env_id),
                "action_norm_mean": self._episode_mean(self._episode_action_norm_sum, env_id),
                "action_rate_mean": self._episode_mean(self._episode_action_rate_sum, env_id),
                "delta_x_des_H_mean": self._episode_mean(self._episode_delta_x_des_sum, env_id),
                "delta_x_meas_H_mean": self._episode_mean(self._episode_delta_x_meas_sum, env_id),
            }
            for name in self.DONE_CAUSES:
                row[name] = int(bool(cause_masks[name][env_id].detach().cpu().item()))
            rows.append(row)
        self._append_csv(rows)
        self._episode_records.extend(rows)
        self._write_reports()

    def _append_csv(self, rows: list[dict[str, float | int | str]]) -> None:
        fieldnames = list(rows[0].keys())
        write_header = (not self._csv_header_written) or (not self._diagnostic_csv.exists())
        with self._diagnostic_csv.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        self._csv_header_written = True

    def _summarize_records(self) -> dict[str, float]:
        if not self._episode_records:
            return {}
        n = float(len(self._episode_records))
        summary: dict[str, float] = {"episodes": n}
        for name in self.DONE_CAUSES:
            summary[f"{name}_rate"] = sum(float(r[name]) for r in self._episode_records) / n
            summary[f"{name}_count"] = sum(float(r[name]) for r in self._episode_records)
        numeric = (
            "episode_length",
            "episode_return",
            "final_ee_error",
            "ee_error_improvement",
            "guard_fraction",
            "clipping_fraction",
            "hydraulic_consistency_error",
            "action_norm_mean",
        )
        for key in numeric:
            summary[f"mean_{key}"] = sum(float(r[key]) for r in self._episode_records) / n
        early_fail = [
            r for r in self._episode_records
            if int(r["episode_length"]) < 20 and not bool(r["success"])
        ]
        summary["early_failure_rate"] = len(early_fail) / n
        window = max(1, len(self._episode_records) // 5)
        for prefix, records in (("first_20pct", self._episode_records[:window]), ("last_20pct", self._episode_records[-window:])):
            denom = float(len(records))
            for key in ("episode_return", "episode_length", "final_ee_error", "ee_error_improvement"):
                summary[f"{prefix}_mean_{key}"] = sum(float(r[key]) for r in records) / denom
            summary[f"{prefix}_early_failure_rate"] = (
                sum(1 for r in records if int(r["episode_length"]) < 20 and not bool(r["success"])) / denom
            )
        return summary

    def _write_reports(self) -> None:
        summary = self._summarize_records()
        if not summary:
            return
        lines = [
            "# PPO Horizon Termination Diagnostics",
            "",
            f"- episodes: {int(summary['episodes'])}",
            f"- mean_episode_length: {summary['mean_episode_length']:.3f}",
            f"- mean_episode_return: {summary['mean_episode_return']:.3f}",
            f"- mean_final_ee_error: {summary['mean_final_ee_error']:.6f}",
            f"- mean_ee_error_improvement: {summary['mean_ee_error_improvement']:.6f}",
            f"- early_failure_rate: {summary['early_failure_rate']:.6f}",
            "",
            "## Termination Cause Histogram",
            "",
        ]
        for name in self.DONE_CAUSES:
            lines.append(f"- {name}: {int(summary[f'{name}_count'])} ({summary[f'{name}_rate']:.3f})")
        lines.extend(
            [
                "",
                "## Tracking And Safety Means",
                "",
                f"- mean_guard_fraction: {summary['mean_guard_fraction']:.6f}",
                f"- mean_clipping_fraction: {summary['mean_clipping_fraction']:.6f}",
                f"- mean_hydraulic_error: {summary['mean_hydraulic_consistency_error']:.6f}",
                f"- mean_action_norm: {summary['mean_action_norm_mean']:.6f}",
            ]
        )
        self._diagnostic_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        analysis = [
            "# PPO Iteration Termination Analysis",
            "",
            "This report is updated online from completed horizon episodes.",
            "",
            "## Reward Trend",
            f"- mean_episode_return: {summary['mean_episode_return']:.3f}",
            f"- first_20pct_mean_episode_return: {summary['first_20pct_mean_episode_return']:.3f}",
            f"- last_20pct_mean_episode_return: {summary['last_20pct_mean_episode_return']:.3f}",
            "",
            "## Episode Length Trend",
            f"- mean_episode_length: {summary['mean_episode_length']:.3f}",
            f"- first_20pct_mean_episode_length: {summary['first_20pct_mean_episode_length']:.3f}",
            f"- last_20pct_mean_episode_length: {summary['last_20pct_mean_episode_length']:.3f}",
            "",
            "## Termination Cause Histogram",
        ]
        for name in self.DONE_CAUSES:
            analysis.append(f"- {name}: {int(summary[f'{name}_count'])} ({summary[f'{name}_rate']:.3f})")
        failure_rate = 1.0 - summary["success_rate"] - summary["timeout_rate"]
        verdict = "inconclusive"
        if summary["early_failure_rate"] > 0.5 and summary["success_rate"] < 0.1:
            verdict = "likely exploiting early failure termination"
        elif summary["success_rate"] > 0.0 and summary["mean_final_ee_error"] < 0.05:
            verdict = "showing real strict tracking successes"
        analysis.extend(
            [
                "",
                "## Rates",
                f"- success_rate: {summary['success_rate']:.6f}",
                f"- failure_rate: {failure_rate:.6f}",
                f"- timeout_rate: {summary['timeout_rate']:.6f}",
                f"- early_termination_rate: {summary['early_failure_rate']:.6f}",
                f"- first_20pct_early_failure_rate: {summary['first_20pct_early_failure_rate']:.6f}",
                f"- last_20pct_early_failure_rate: {summary['last_20pct_early_failure_rate']:.6f}",
                "",
                "## Final EE Error Trend",
                f"- mean_final_ee_error: {summary['mean_final_ee_error']:.6f}",
                f"- mean_ee_error_improvement: {summary['mean_ee_error_improvement']:.6f}",
                f"- first_20pct_mean_final_ee_error: {summary['first_20pct_mean_final_ee_error']:.6f}",
                f"- last_20pct_mean_final_ee_error: {summary['last_20pct_mean_final_ee_error']:.6f}",
                f"- first_20pct_mean_ee_error_improvement: {summary['first_20pct_mean_ee_error_improvement']:.6f}",
                f"- last_20pct_mean_ee_error_improvement: {summary['last_20pct_mean_ee_error_improvement']:.6f}",
                "",
                "## Verdict",
                f"- {verdict}",
                "",
                "## Required Reward/Termination Fixes",
                "- Failure terminations now receive a strong negative reward.",
                "- Strict success requires low final EE error, sustained hold, no active guard, and a minimum episode length.",
                "- Cab, workspace, cylinder, joint, nonfinite, and base-flying events are logged separately.",
            ]
        )
        self._analysis_report.write_text("\n".join(analysis) + "\n", encoding="utf-8")

    def _build_log(self, done: torch.Tensor, cause_masks: dict[str, torch.Tensor], final_ee_error: torch.Tensor) -> dict[str, torch.Tensor]:
        done_f = done.float()
        denom = done_f.sum().clamp(min=1.0)
        failure = done.clone()
        failure &= ~cause_masks["success"]
        failure &= ~cause_masks["timeout"]
        early = done & (self._horizon_episode_length < 20) & ~cause_masks["success"]
        return {
            "Diagnostics/mean_final_ee_error": final_ee_error.detach().mean(),
            "Diagnostics/mean_ee_error_improvement_per_horizon": self._last_reward_terms.get("progress", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/success_rate": (cause_masks["success"].float().sum() / denom).detach(),
            "Diagnostics/failure_rate": (failure.float().sum() / denom).detach(),
            "Diagnostics/timeout_rate": (cause_masks["timeout"].float().sum() / denom).detach(),
            "Diagnostics/early_termination_rate": (early.float().sum() / denom).detach(),
            "Diagnostics/mean_episode_length": self._horizon_episode_length.float().mean().detach(),
            "Diagnostics/mean_guard_fraction": self._last_reward_terms.get("guard_fraction", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/mean_clipping_fraction": self._last_reward_terms.get("clipping_fraction", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/mean_hydraulic_error": self._last_reward_terms.get("hydraulic_consistency", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/mean_action_norm": self._prev_action.norm(dim=-1).mean().detach(),
            "Diagnostics/reward_progress": self._last_reward_terms.get("progress", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/reward_proximity": self._last_reward_terms.get("proximity", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/reward_ee_error": self._last_reward_terms.get("ee_error_after", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/reward_action_rate": self._last_reward_terms.get("action_rate", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/reward_hydraulic": self._last_reward_terms.get("hydraulic_consistency", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/reward_guard": self._last_reward_terms.get("guard_fraction", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/reward_clip": self._last_reward_terms.get("clipping_fraction", torch.zeros_like(final_ee_error)).mean(),
            "Diagnostics/reward_do_no_harm": self._last_reward_terms.get("do_no_harm", torch.zeros_like(final_ee_error)).mean(),
        }

    def step(self, action: torch.Tensor):
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = torch.clamp(action, -1.0, 1.0)
        action = self._coerce_action_dim(action)
        prev_action = self._prev_action.detach().clone()

        # -- Task-space: map EE action to cylinder delta via Jacobian --
        if self._is_task_space:
            desired_ee_delta = self._expand_task_action(action) * self._ee_action_scale
            # Neutral detection: small actions are treated as hold
            neutral_action = action.abs().amax(dim=-1) <= float(getattr(self.cfg, "horizon_zero_action_deadband", 1.0e-5))
            # Map to cylinder space
            delta_x_des_H, _ = self._ee_controller.map_ee_to_cylinder(desired_ee_delta)
            delta_x_des_H = torch.where(neutral_action.unsqueeze(-1), torch.zeros_like(delta_x_des_H), delta_x_des_H)
        else:
            # -- Cylinder-space: action directly maps to cylinder displacement --
            delta_x_des_H = action * self._action_scale
            neutral_action = action.abs().amax(dim=-1) <= float(getattr(self.cfg, "horizon_zero_action_deadband", 1.0e-5))
            delta_x_des_H = torch.where(neutral_action.unsqueeze(-1), torch.zeros_like(delta_x_des_H), delta_x_des_H)


        error_before = compute_tracking_error(self._uw).detach()
        cq0 = self._uw.robot.data.joint_pos[:, self._uw._cylinder_joint_ids].detach().clone()
        x0, xd0 = self._uw._cylinder_adapter.current_from_cylinder_joints(
            cq0,
            self._uw.robot.data.joint_vel[:, self._uw._cylinder_joint_ids],
        )

        delta_fd_H = self._idfd_rollout(delta_x_des_H, x0, xd0)
        delta_fd_H = torch.where(neutral_action.unsqueeze(-1), torch.zeros_like(delta_fd_H), delta_fd_H)
        delta_cmd_raw_H = self._fd_gains * delta_fd_H
        clip_m = float(getattr(self.cfg, "horizon_delta_cmd_clip_m", 0.02))
        delta_cmd_H = torch.clamp(delta_cmd_raw_H, -clip_m, clip_m)
        clipping_fraction = (delta_cmd_H - delta_cmd_raw_H).abs().gt(1.0e-8).float().mean(dim=-1)
        delta_cmd_H = torch.where(neutral_action.unsqueeze(-1), torch.zeros_like(delta_cmd_H), delta_cmd_H)

        final_target, guard_flags, terminated, truncated = self._smooth_ramp(delta_cmd_H, cq0, neutral_action)
        base_masks = self._base_done_masks()

        cq1 = self._uw.robot.data.joint_pos[:, self._uw._cylinder_joint_ids].detach().clone()
        x1, _ = self._uw._cylinder_adapter.current_from_cylinder_joints(
            cq1,
            self._uw.robot.data.joint_vel[:, self._uw._cylinder_joint_ids],
        )
        delta_x_meas_H = x1 - x0
        error_after = compute_tracking_error(self._uw).detach()

        # -- Determine stage before reward computation --
        stage0 = self._is_stage0()
        soft_ws = bool(getattr(self.cfg, "workspace_violation_soft", False)) and stage0

        if self._is_task_space:
            reward, terms = compute_high_level_reward(
                self._uw, action, prev_action, error_before, error_after,
            )
        elif stage0:
            reward, terms = compute_stage0_reward(
                self._uw, action, prev_action, error_before, error_after,
                delta_cmd_H, delta_x_meas_H, guard_flags, clipping_fraction,
            )
        else:
            reward, terms = compute_horizon_reward(
                self._uw,
                action,
                prev_action,
                error_before,
                error_after,
                delta_cmd_H,
                delta_x_meas_H,
                guard_flags,
            )
        guard_fraction = terms["guard_fraction"]
        hydraulic_error = terms["hydraulic_consistency"]

        workspace_violation, cab_violation = self._workspace_guard_components()
        cylinder_limit = guard_flags.bool().any(dim=-1) & ~(workspace_violation | cab_violation)
        nonfinite = ~(
            torch.isfinite(action).all(dim=-1)
            & torch.isfinite(delta_fd_H).all(dim=-1)
            & torch.isfinite(delta_cmd_H).all(dim=-1)
            & torch.isfinite(delta_x_meas_H).all(dim=-1)
            & torch.isfinite(error_after)
        )
        ground_hard_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        if self._trajectory_mode.startswith("ground_"):
            ground_metrics_now = self.ground_safety_metrics()
            target_bad = ground_metrics_now["target_tool_bottom_clearance"] < (self._tool_ground_clearance - 0.005)
            ee_bad = ground_metrics_now["tool_bottom_clearance"] < -0.01
            arm_bad = ground_metrics_now["arm_geometry_clearance"] < -0.02
            huge_bad = ground_metrics_now["max_abs_body_world_position"] > 100.0
            root_bad = ground_metrics_now["root_motion_norm"] > 100.0
            finite_bad = ground_metrics_now["nan_inf_count"] > 0
            ground_hard_fail = target_bad | ee_bad | arm_bad | huge_bad | root_bad | finite_bad

        # -- Stage-0: track consecutive workspace violations --
        ws_consecutive_threshold = int(getattr(self.cfg, "workspace_violation_done_after_consecutive_steps", 10))
        ws_penalty = float(getattr(self.cfg, "workspace_violation_penalty", -3.0))

        if soft_ws:
            self._ws_consecutive_steps = torch.where(
                workspace_violation,
                self._ws_consecutive_steps + 1,
                torch.zeros_like(self._ws_consecutive_steps),
            )
            ws_hard_done = self._ws_consecutive_steps >= ws_consecutive_threshold
        else:
            ws_hard_done = workspace_violation

        severe_cab = bool(getattr(self.cfg, "severe_cab_collision_done", True))
        nonfinite_done_flag = bool(getattr(self.cfg, "nonfinite_done", True))
        failure_done = (
            terminated
            | (cab_violation if severe_cab else terminated & False)
            | ws_hard_done
            | cylinder_limit
            | (nonfinite if nonfinite_done_flag else terminated & False)
            | ground_hard_fail
        )

        # -- Stage-0: relaxed success criteria (level-based) --
        if stage0:
            level = int(getattr(self.cfg, "stage0_level", 0))
            if level == 0:
                success_tolerance = float(getattr(self.cfg, "stage0_l0_success_tolerance_m", 0.10))
                required_hold_steps = int(getattr(self.cfg, "stage0_l0_required_hold_steps", 3))
                min_success_steps = int(getattr(self.cfg, "stage0_l0_min_episode_steps", 10))
            elif level == 1:
                success_tolerance = float(getattr(self.cfg, "stage0_l1_success_tolerance_m", 0.08))
                required_hold_steps = int(getattr(self.cfg, "stage0_l1_required_hold_steps", 5))
                min_success_steps = int(getattr(self.cfg, "stage0_l1_min_episode_steps", 20))
            else:
                success_tolerance = float(getattr(self.cfg, "stage0_l2_success_tolerance_m", 0.08))
                required_hold_steps = int(getattr(self.cfg, "stage0_l2_required_hold_steps", 5))
                min_success_steps = int(getattr(self.cfg, "stage0_l2_min_episode_steps", 20))
            max_steps = int(getattr(self.cfg, "stage0_max_episode_steps", 100))
        else:
            success_tolerance = float(getattr(self.cfg, "horizon_success_tolerance_m", 0.05))
            required_hold_steps = int(getattr(self.cfg, "horizon_success_required_hold_steps", 10))
            min_success_steps = int(getattr(self.cfg, "horizon_min_success_steps", 30))
            max_steps = None
        tracking_ok = error_after < success_tolerance
        guard_active = cab_violation | workspace_violation | cylinder_limit | (guard_fraction > 0.0)
        hold_ok = tracking_ok & ~guard_active & ~failure_done
        self._success_hold_steps = torch.where(
            hold_ok,
            self._success_hold_steps + 1,
            torch.zeros_like(self._success_hold_steps),
        )
        strict_success = (
            (self._success_hold_steps >= required_hold_steps)
            & (self._horizon_episode_length + 1 > min_success_steps)
            & ~guard_active
            & ~failure_done
        )

        # -- Stage-0: soft penalty for workspace violation --
        if soft_ws:
            rew_ws_penalty = ws_penalty * workspace_violation.float()
        else:
            rew_ws_penalty = torch.zeros_like(reward)

        failure_penalty = float(getattr(self.cfg, "horizon_failure_termination_penalty", 20.0))
        success_bonus = float(getattr(self.cfg, "horizon_success_bonus", 20.0))
        reward = reward - failure_penalty * failure_done.float() + rew_ws_penalty
        reward = reward + success_bonus * strict_success.float()
        reward = torch.nan_to_num(reward, nan=-failure_penalty, posinf=0.0, neginf=-failure_penalty)
        self._last_reward_terms = terms
        self._last_reward_terms["clipping_fraction"] = clipping_fraction.detach()

        # -- Stage-0: timeout at max episode steps --
        if max_steps is not None:
            env_truncated = self._horizon_episode_length + 1 >= max_steps
        else:
            env_truncated = truncated.clone()

        truncated = env_truncated & ~failure_done & ~strict_success
        terminated = failure_done | strict_success

        done = terminated | truncated

        cause_masks = {name: torch.zeros(self.num_envs, dtype=torch.bool, device=self._device) for name in self.DONE_CAUSES}
        cause_masks["success"] = strict_success
        cause_masks["timeout"] = base_masks["timeout"] | (truncated & ~cause_masks["success"])
        cause_masks["cab_guard_violation"] = cab_violation & done
        cause_masks["workspace_violation"] = (workspace_violation | base_masks["workspace_violation"]) & done
        cause_masks["joint_limit"] = base_masks["joint_limit"] & done
        cause_masks["cylinder_limit"] = cylinder_limit & done
        cause_masks["ground_safety"] = ground_hard_fail & done
        cause_masks["NaN/nonfinite"] = (nonfinite | base_masks["NaN/nonfinite"]) & done
        cause_masks["base_flying"] = base_masks["base_flying"] & done
        known = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        for mask in cause_masks.values():
            known |= mask
        cause_masks["unknown_done"] = done & ~known

        self._update_episode_sums(
            reward,
            action,
            prev_action,
            delta_x_des_H,
            delta_x_meas_H,
            guard_fraction,
            clipping_fraction,
            hydraulic_error,
        )
        log_info = self._build_log(done, cause_masks, error_after)
        if self._trajectory_mode.startswith("ground_"):
            target_world = self.current_ground_target_world()
            ee_tool_world = self._ee_or_tool_world_pos()
            ground_metrics = self.ground_safety_metrics(target_world)
            log_info.update({
                "Trajectory/current_mode": torch.tensor(0.0, device=self._device),
                "Trajectory/ground_z": ground_metrics["ground_z"].detach().mean(),
                "Trajectory/target_world_x": target_world[:, 0].detach().mean(),
                "Trajectory/target_world_y": target_world[:, 1].detach().mean(),
                "Trajectory/target_world_z": target_world[:, 2].detach().mean(),
                "Trajectory/ee_origin_world_z": ground_metrics["ee_origin_world_z"].detach().mean(),
                "Trajectory/tool_or_ee_world_x": ee_tool_world[:, 0].detach().mean(),
                "Trajectory/tool_or_ee_world_y": ee_tool_world[:, 1].detach().mean(),
                "Trajectory/tool_or_ee_world_z": ee_tool_world[:, 2].detach().mean(),
                "Trajectory/tool_bottom_world_z": ground_metrics["tool_bottom_world_z"].detach().mean(),
                "Trajectory/target_tool_bottom_world_z": ground_metrics["target_tool_bottom_world_z"].detach().mean(),
                "Trajectory/tracking_error": ground_metrics["tracking_error"].detach().mean(),
                "Trajectory/min_target_ground_clearance": ground_metrics["target_tool_bottom_clearance"].detach().min(),
                "Trajectory/min_ee_ground_clearance": ground_metrics["tool_bottom_clearance"].detach().min(),
                "Trajectory/min_arm_link_ground_clearance": ground_metrics["arm_geometry_clearance"].detach().min(),
                "Trajectory/tool_bottom_clearance": ground_metrics["tool_bottom_clearance"].detach().mean(),
                "Trajectory/arm_geometry_clearance": ground_metrics["arm_geometry_clearance"].detach().mean(),
                "Trajectory/max_abs_body_world_position": ground_metrics["max_abs_body_world_position"].detach().max(),
                "Trajectory/nan_inf_count": ground_metrics["nan_inf_count"].detach().sum().float(),
                "Trajectory/huge_transform_warning_count": (
                    ground_metrics["max_abs_body_world_position"] > 100.0
                ).detach().sum().float(),
                "Trajectory/cylinder_limit_fraction": ground_metrics["cylinder_limit_fraction"].detach().mean(),
                "Trajectory/path_completion_fraction": ground_metrics["path_completion_fraction"].detach().mean(),
                "Trajectory/root_motion_norm": ground_metrics["root_motion_norm"].detach().max(),
            })
        self._record_done_episodes(done, cause_masks, reward, error_after)

        self._prev_action = action.detach().clone()
        self._prev_hl_action = action.detach().clone()
        self._prev_delta_x_des = delta_x_des_H.detach().clone()
        self._prev_fd_delta = delta_fd_H.detach().clone()
        self._prev_meas_delta = delta_x_meas_H.detach().clone()
        self._prev_cmd_delta = delta_cmd_H.detach().clone()
        self._prev_guard_flags = guard_flags.detach().clone()
        self._cylinder_hold_target = final_target.detach().clone()

        done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            self._uw._reset_idx(done_env_ids)
            self._prev_action[done_env_ids] = 0.0
            self._prev_hl_action[done_env_ids] = 0.0
            self._prev_delta_x_des[done_env_ids] = 0.0
            self._prev_fd_delta[done_env_ids] = 0.0
            self._prev_meas_delta[done_env_ids] = 0.0
            self._prev_cmd_delta[done_env_ids] = 0.0
            self._prev_guard_flags[done_env_ids] = 0.0
            self._refresh_root_hold_from_current()
            self._refresh_hold_target_from_current()
            self._reset_episode_buffers(done_env_ids)
            if self._is_stage0():
                self._resample_stage0_target(done_env_ids)
            if self._trajectory_mode.startswith("ground_") and self._ground_path_world is not None:
                self.set_ground_target_world(self._ground_path_world[self._ground_path_index])

        if self._trajectory_mode.startswith("ground_"):
            self.advance_ground_trajectory(1)
            self._update_ground_visualization()

        extras = {
            "guard_fraction": self._last_guard_fraction.detach().clone(),
            "reward_terms": self._last_reward_terms,
            "log": log_info,
        }
        return self._build_observation(), reward, terminated, truncated, extras

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
        self.num_obs = env.observation_space
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
