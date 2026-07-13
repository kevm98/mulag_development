"""Deterministic per-environment stop-and-mow state machine.

Transitions are based on measured physical state (truck speed, cylinder speeds,
cutter height, path progress), not only elapsed steps.
"""

from __future__ import annotations

import torch

from .parking import ParkingBrake
from .phases import StopMowPhase, StopMowState


class StopMowStateMachine:
    """Deterministic state machine for arm-only stop-and-mow cycles.

    The truck is parked during all arm work.  The state machine advances each
    environment through RESET_SETTLE -> DEPLOY -> MOW_OUT <-> MOW_RETURN ->
    STOW -> COMPLETE.  Multi-stroke accumulation repeats MOW_OUT/MOW_RETURN
    until the target accumulated path length is reached.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        brake: ParkingBrake,
        *,
        reset_settle_steps: int = 300,
        reset_settle_duration_s: float = 3.0,
        deploy_hold_steps: int = 60,
        min_accumulated_path_m: float = 5.0,
        stow_target_height_m: float = 0.30,
        stow_steps: int = 300,
        step_dt: float = 1.0 / 60.0,
    ):
        self.num_envs = int(num_envs)
        self.device = device
        self.brake = brake
        self.state = StopMowState(num_envs, device)

        self.reset_settle_steps = int(reset_settle_steps)
        self.reset_settle_duration_s = float(reset_settle_duration_s)
        self.deploy_hold_steps = int(deploy_hold_steps)
        self.required_half_strokes = int(float(min_accumulated_path_m))
        self.stow_target_height_m = float(stow_target_height_m)
        self.stow_steps = int(stow_steps)
        self.step_dt = float(step_dt)

        # Tolerances
        self.truck_stop_threshold_ms = 0.02
        self.cylinder_speed_threshold = 0.005  # m/s (learned x space)
        self.deploy_error_tolerance_m = 0.05
        self.stow_height_tolerance_m = 0.05

    @torch.no_grad()
    def step(
        self,
        cutter_height: torch.Tensor,
        cylinder_velocity: torch.Tensor,
        normalized_progress_delta: torch.Tensor,
    ) -> None:
        """Advance the state machine by one control step.

        Args:
            cutter_height: world-Z of the cutter bottom ``(N,)`` [m].
            cylinder_velocity: learned cylinder velocities ``(N, 4)`` [m/s].
            normalized_progress_delta: normalized progress increment ``(N,)`` [0,1].
        """
        N = self.num_envs
        dt = self.step_dt

        # Update per-phase counters
        self.state.phase_step += 1
        self.state.phase_time += dt

        truck_stopped = self.brake.truck_is_stopped(self.truck_stop_threshold_ms)
        cyl_slow = (cylinder_velocity.abs().max(dim=-1).values < self.cylinder_speed_threshold)

        # Phase: RESET_SETTLE
        in_reset = self.state.phase == int(StopMowPhase.RESET_SETTLE)
        reset_ok = in_reset & (
            (self.state.phase_step >= self.reset_settle_steps)
            | ((self.state.phase_time >= self.reset_settle_duration_s) & truck_stopped & cyl_slow)
        )
        if reset_ok.any():
            ids = torch.where(reset_ok)[0]
            self.state.to_phase(StopMowPhase.DEPLOY, ids)

        # Phase: DEPLOY
        in_deploy = self.state.phase == int(StopMowPhase.DEPLOY)
        deploy_done = in_deploy & (self.state.phase_step >= self.deploy_hold_steps) & truck_stopped
        if deploy_done.any():
            ids = torch.where(deploy_done)[0]
            self.state.to_phase(StopMowPhase.MOW_OUT, ids)

        # Phase: MOW_OUT -> MOW_RETURN
        in_mow_out = self.state.phase == int(StopMowPhase.MOW_OUT)
        mow_out_done = in_mow_out & (self.state.path_progress >= 1.0)
        if mow_out_done.any():
            ids = torch.where(mow_out_done)[0]
            self.state.path_progress[ids] = 0.0
            self.state.stroke_index[ids] += 1
            self.state.total_strokes[ids] += 1
            self.state.to_phase(StopMowPhase.MOW_RETURN, ids)

        # Phase: MOW_RETURN -> MOW_OUT (repeat) or STOW
        in_mow_return = self.state.phase == int(StopMowPhase.MOW_RETURN)
        mow_return_done = in_mow_return & (self.state.path_progress >= 1.0)
        if mow_return_done.any():
            ids = torch.where(mow_return_done)[0]
            self.state.path_progress[ids] = 0.0
            self.state.stroke_index[ids] += 1
            self.state.total_strokes[ids] += 1
            # If accumulated enough, go to STOW; otherwise repeat
            enough = self.state.accumulated_path[ids] >= self.required_half_strokes
            stow_ids = ids[enough]
            repeat_ids = ids[~enough]
            if len(stow_ids) > 0:
                self.state.to_phase(StopMowPhase.STOW, stow_ids)
            if len(repeat_ids) > 0:
                self.state.to_phase(StopMowPhase.MOW_OUT, repeat_ids)

        # Phase: STOW -> COMPLETE
        in_stow = self.state.phase == int(StopMowPhase.STOW)
        stow_done = in_stow & (self.state.phase_step >= self.stow_steps) & truck_stopped
        if stow_done.any():
            ids = torch.where(stow_done)[0]
            self.state.to_phase(StopMowPhase.COMPLETE, ids)

        # Accumulate path progress for active mowing environments
        active_mow = torch.logical_or(
            self.state.phase == int(StopMowPhase.MOW_OUT),
            self.state.phase == int(StopMowPhase.MOW_RETURN),
        )
        if active_mow.any():
            ids = torch.where(active_mow)[0]
            self.state.path_progress[ids] += normalized_progress_delta[ids]
            self.state.accumulated_path[ids] += normalized_progress_delta[ids]

    def is_mowing(self) -> torch.Tensor:
        """Return ``True`` for envs currently in a mowing phase (MOW_OUT or MOW_RETURN)."""
        p = self.state.phase
        return (p == int(StopMowPhase.MOW_OUT)) | (p == int(StopMowPhase.MOW_RETURN))

    def is_settling(self) -> torch.Tensor:
        """Return ``True`` for envs in reset-settle or deploy."""
        p = self.state.phase
        return (p == int(StopMowPhase.RESET_SETTLE)) | (p == int(StopMowPhase.DEPLOY))

    def is_complete(self) -> torch.Tensor:
        """Return ``True`` for envs that have finished all cycles."""
        return self.state.phase == int(StopMowPhase.COMPLETE)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the state machine for selected environments."""
        self.state.reset(env_ids)
