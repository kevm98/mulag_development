"""Parking brake management for the MULAG truck.

Applies high-stiffness position hold to prevent truck motion during arm work.
"""

from __future__ import annotations

import torch


class ParkingBrake:
    """Manage the wheel parking brake via joint stiffness/damping toggling."""

    def __init__(self, env):
        self._env = env
        self._wheel_ids = env._wheel_joint_ids
        self._parked_stiffness = 2.0e9
        self._parked_damping = 1.0e7
        self._drive_stiffness = 0.0
        self._drive_damping = 1.0e6

    @torch.no_grad()
    def engage(self) -> None:
        """Lock the wheels at their current positions."""
        env = self._env
        env.robot.write_joint_stiffness_to_sim(self._parked_stiffness, joint_ids=self._wheel_ids)
        env.robot.write_joint_damping_to_sim(self._parked_damping, joint_ids=self._wheel_ids)
        cur = env.robot.data.joint_pos[:, self._wheel_ids].detach().clone()
        env.robot.set_joint_position_target(cur, joint_ids=self._wheel_ids)

    @torch.no_grad()
    def release(self) -> None:
        """Switch wheels to velocity-drive mode."""
        self._env.robot.write_joint_stiffness_to_sim(self._drive_stiffness, joint_ids=self._wheel_ids)
        self._env.robot.write_joint_damping_to_sim(self._drive_damping, joint_ids=self._wheel_ids)

    @torch.no_grad()
    def truck_speed_ms(self) -> torch.Tensor:
        """Current truck chassis speed magnitude ``(N,)`` [m/s]."""
        vel = self._env.robot.data.root_lin_vel_w[:, :2]
        return torch.linalg.norm(vel, dim=-1)

    @torch.no_grad()
    def truck_is_stopped(self, threshold_ms: float = 0.02) -> torch.Tensor:
        """Return ``True`` for envs where truck speed is below *threshold_ms*."""
        return self.truck_speed_ms() < float(threshold_ms)
