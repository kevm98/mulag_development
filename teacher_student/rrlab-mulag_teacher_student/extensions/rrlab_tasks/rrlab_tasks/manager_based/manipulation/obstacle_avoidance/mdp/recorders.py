from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import Optional

from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_apply_inverse, euler_xyz_from_quat
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation

class PostStepNormalizedJointPositionActionsRecorder(RecorderTerm):
    """Recorder term that records the proceseed joint position actions which send to the env,
       then the processed action will be normalized to [-1, 1] based on the joint limits then recorded.
    """

    def record_post_step(self):
        processed_actions: Optional[torch.Tensor] = None

        # Loop through active terms and concatenate their processed actions
        for term_name in self._env.action_manager.active_terms:
            term_actions = self._env.action_manager.get_term(term_name).processed_actions.clone()
            if processed_actions is None:
                processed_actions = term_actions
            else:
                processed_actions = torch.cat([processed_actions, term_actions], dim=-1)

        return "actions", self.normalize_actions(processed_actions)
    

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize the actions to [-1, 1] based on joint limits.

        Args:
            actions (torch.Tensor): The processed joint position actions.

        Returns:
            torch.Tensor: The normalized joint position actions.
        """
        joint_ids = self._env.action_manager.get_term("arm_action")._joint_ids  # type: ignore
        lower_limits = self._env.scene["robot"].data.soft_joint_pos_limits[:, joint_ids, 0]
        upper_limits = self._env.scene["robot"].data.soft_joint_pos_limits[:, joint_ids, 1]

        # Normalize actions to [-1, 1]
        normalized_actions = 2 * (actions - lower_limits) / (upper_limits - lower_limits) - 1
        normalized_actions = torch.clamp(normalized_actions, -1.0, 1.0)

        return normalized_actions
    
class PreStepStudentObservationsRecorder(RecorderTerm):
    """Recorder term that records a filtered subset of the policy group observations each step."""

    # keep as a constant list so it’s obvious what you record
    KEEP_KEYS = (
        "tcp_height",
        "joint_pos",
        "joint_vel",
        "distance_to_goal",
        "last_action",
        "unimog_pose",
        "unimog_pose_dot",
        "camera",
    )

    def record_pre_step(self):
        policy_obs = self._env.obs_buf["policy"]  # dict[str, torch.Tensor] (or numpy after wrappers)
        filtered = {k: policy_obs[k] for k in self.KEEP_KEYS if k in policy_obs}
        return "obs", filtered


class ExperimentInfoRecorder(RecorderTerm):
    """Recorder term that records per-env experiment info each step.

    IMPORTANT:
    - Every returned metric is indexed by env_id, i.e. first dim == num_envs.
    - Scalars are avoided (no torch.mean over env dimension).
    """

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        # Robot and sensors
        self.robot: Articulation = self._env.scene["robot"]
        self.body_name = "Messerkopf"
        self.body_idx = self.robot.find_bodies(self.body_name)[0]

        self.sensor_left = self._env.scene["height_sensor_left"]
        self.sensor_right = self._env.scene["height_sensor_right"]
        self.tcp_transformer = self._env.scene["tcp_transformer"]
        self.contact_sensor = self._env.scene["contact_sensor_head"]
        self.distance_scanner = self._env.scene["distance_scanner"]

    def record_post_step(self):
        metrics: dict[str, torch.Tensor] = {}

        N = self._env.num_envs
        device = self._env.device

        # ----------------------------------------------------------------------
        # 1) Lateral deviation (PER ENV) -> shape [N]
        # ----------------------------------------------------------------------
        env_origins = self._env.scene.env_origins               # [N,3]
        root_positions = self.robot.data.root_pos_w             # [N,3]
        dev_all = torch.abs(torch.abs(env_origins[:, 1] - root_positions[:, 1]) - 4.3)  # [N]
        metrics["lateral_deviation"] = dev_all

        # ----------------------------------------------------------------------
        # 2) Force magnitude in TCP frame (PER ENV) -> shape [N]
        # ----------------------------------------------------------------------
        forces_w = self.contact_sensor.data.net_forces_w        # likely [N, B, 3] or [N,3]
        rotated_forces = quat_apply_inverse(self.tcp_transformer.data.target_quat_w, forces_w)

        force_xy = torch.norm(rotated_forces[..., :2], dim=-1)  # [N,B] or [N]
        if force_xy.dim() > 1:
            force_xy = torch.max(force_xy, dim=1)[0]            # [N]
        metrics["force_xy_magnitude"] = force_xy

        # ----------------------------------------------------------------------
        # 3) Raycaster height around TCP (PER ENV) -> shape [N]
        # ----------------------------------------------------------------------
        tcp_data = self.tcp_transformer.data                    # contains target_pos_w, target_quat_w, source_pos_w, source_quat_w

        vec_left = self.sensor_left.data.ray_hits_w - tcp_data.target_pos_w     # [N,R,3] - [N,3] (broadcast)
        hits_left_local = quat_apply_inverse(tcp_data.target_quat_w, vec_left)  # [N,R,3]

        vec_right = self.sensor_right.data.ray_hits_w - tcp_data.target_pos_w
        hits_right_local = quat_apply_inverse(tcp_data.target_quat_w, vec_right)

        # mean over rays only -> stays [N]
        avg_h = (torch.mean(hits_left_local[..., 2], dim=-1) + torch.mean(hits_right_local[..., 2], dim=-1)) / 2.0  # [N]
        metrics["avg_raycaster_height"] = torch.clamp(avg_h, min=0.0)

        # ----------------------------------------------------------------------
        # 4) TCP tracking errors (PER ENV) -> shape [N]
        # ----------------------------------------------------------------------
        command = self._env.command_manager.get_command("ee_pose")  # [N,7] (pos xyz + quat)
        des_pos_w, des_quat_w = combine_frame_transforms(
            self.robot.data.root_link_state_w[:, :3],
            self.robot.data.root_link_state_w[:, 3:7],
            command[:, :3],
            command[:, 3:7],
        )

        curr_pos_w = self.robot.data.body_link_state_w[:, self.body_idx[0], :3]   # [N,3]
        curr_quat_w = self.robot.data.body_link_state_w[:, self.body_idx[0], 3:7] # [N,4]

        metrics["tracking_error_pos"] = torch.norm(curr_pos_w[:, :2] - des_pos_w[:, :2], dim=-1)  # [N]
        metrics["tracking_error_rot"] = quat_error_magnitude(curr_quat_w, des_quat_w)             # [N]

        # ----------------------------------------------------------------------
        # 5) Action rate (PER ENV) -> shape [N]
        # ----------------------------------------------------------------------
        curr_action = self._env.action_manager.action            # [N, A] typically
        prev_action = self._env.action_manager.prev_action       # [N, A] or None

        if prev_action is not None:
            metrics["action_rate_l2"] = torch.norm(curr_action - prev_action, dim=-1)  # [N]
        else:
            metrics["action_rate_l2"] = torch.zeros((N,), device=device)

        # ----------------------------------------------------------------------
        # 6) Goal points (PER ENV) -> shape [N,2]
        # ----------------------------------------------------------------------
        # metrics["goal_points"] = command[:, :2]  # [N,2]
        metrics["goal_points"] = des_pos_w[:, :2]  # [N,2]

        # ----------------------------------------------------------------------
        # 7) TCP position in source frame (PER ENV) -> shape [N,2]
        # ----------------------------------------------------------------------
        # world tcp pos - source pos, then rotate into source frame
        # tcp_pos_translated = self.robot.data.body_link_state_w[:, self.body_idx[0], :3] - tcp_data.source_pos_w  # [N,3]
        # tcp_pos_local = quat_apply_inverse(tcp_data.source_quat_w, tcp_pos_translated)                            # [N,3]
        # metrics["tcp_pos"] = tcp_pos_local[:, :2]                                                                # [N,2]

        


        # tcp_pos_local = self.tcp_transformer.data.target_pos_source[:, 0, :] # unimog base coordinate system
        metrics["tcp_pos"] = curr_pos_w[:, :2]      
        # target_xy = command[:, :2]
        # current_xy_local = tcp_pos_local[:, :2]
        # difference_xy = target_xy - current_xy_local # maybe add torch.abs
        # ----------------------------------------------------------------------
        # 8) Yaw angle (PER ENV) -> shape [N]
        # ----------------------------------------------------------------------
        curr_roll, curr_pitch, curr_yaw = euler_xyz_from_quat(curr_quat_w)  # each [N]
        metrics["roll_angle"] = curr_roll
        metrics["pitch_angle"] = curr_pitch
        metrics["yaw_angle"] = curr_yaw

        # ----------------------------------------------------------------------
        # 9) TCP height (PER ENV) -> shape [N]
        # ----------------------------------------------------------------------
        metrics["tcp_height"] = torch.clamp(avg_h, min=0.0)

        # ----------------------------------------------------------------------
        # 10) Contact time (PER ENV) -> shape [N]
        # ----------------------------------------------------------------------
        # Ensure it ends up [N]
        contact_time = self.contact_sensor.data.current_contact_time
        # Some sensors store [N] already; if it stores [N, K], reduce over K deterministically
        if isinstance(contact_time, torch.Tensor) and contact_time.dim() > 1:
            contact_time = torch.max(contact_time, dim=1)[0]
        metrics["contact_time"] = contact_time

        # ----------------------------------------------------------------------
        # 11) Contact force magnitude in TCP frame (PER ENV) -> shape [N]
        # ----------------------------------------------------------------------
        forces_w_all = self.contact_sensor.data.net_forces_w  # [N,B,3] or [N,3]
        tcp_quat_all = tcp_data.target_quat_w                 # [N,4]
        rotated_forces_all = quat_apply_inverse(tcp_quat_all, forces_w_all)

        contact_force = torch.norm(rotated_forces_all, dim=-1)  # [N,B] or [N]
        if contact_force.dim() > 1:
            contact_force = torch.max(contact_force, dim=1)[0]  # [N]
        metrics["contact_force"] = contact_force

        # ----------------------------------------------------------------------
        # 12) Distance to obstacle: min distance over rays (PER ENV) -> shape [N]
        # ----------------------------------------------------------------------
        # ray hits: [N,R,3]
        distance_rays = self.distance_scanner.data.ray_hits_w

        tcp_target_pos = tcp_data.target_pos_w   # [N,3]
        tcp_target_quat = tcp_data.target_quat_w # [N,4]

        R = distance_rays.shape[1]
        translated_hits = distance_rays - tcp_target_pos  # [N,R,3]

        tcp_target_quat_expanded = tcp_target_quat.expand(-1, R, -1)  # [N,R,4]
        hits_local = quat_apply_inverse(tcp_target_quat_expanded, translated_hits)  # [N,R,3]

        invalid = torch.isnan(hits_local).any(dim=-1)  # [N,R]

        dist = torch.norm(hits_local, dim=-1)  # [N,R]
        dist = torch.where(invalid, torch.full_like(dist, 15.0), dist)
        dist = torch.clamp(dist, 0.0, 15.0)

        metrics["distance_to_obstacle"] = dist.min(dim=1).values  # [N]

        # ----------------------------------------------------------------------
        # Safety check: first dim must be N for all tensors
        # ----------------------------------------------------------------------
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 0:
                    raise RuntimeError(f"Metric '{k}' is 0-dim (scalar). Must be per-env with first dim == {N}.")
                if v.shape[0] != N:
                    raise RuntimeError(f"Metric '{k}' has shape {tuple(v.shape)} but expected first dim {N}.")

        # Optional: lightweight debug (don’t .item() env vectors!)
        # print("recorders saving metrics shapes:", {k: tuple(v.shape) for k, v in metrics.items() if isinstance(v, torch.Tensor)})

        return "episode_info", metrics