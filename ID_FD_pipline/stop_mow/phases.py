"""Stop-and-mow phase definitions and per-environment state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import torch


class StopMowPhase(IntEnum):
    """Deterministic phases of the stop-and-mow cycle."""

    RESET_SETTLE = 0   # brake, wait for truck stop, settle cylinders
    DEPLOY = 1         # lower cutter to start pose, verify height + head-down
    MOW_OUT = 2        # outward mowing stroke
    MOW_RETURN = 3     # return mowing stroke (cosine turnaround)
    STOW = 4           # lift cutter to safe travel height
    COMPLETE = 5       # all cycles done


@dataclass
class StopMowState:
    """Per-environment state for the stop-and-mow state machine.

    All tensors are batched ``(num_envs,)`` and live on the environment device.
    """

    num_envs: int
    device: torch.device

    phase: torch.Tensor = field(init=False)           # (N,) int
    phase_step: torch.Tensor = field(init=False)       # (N,) int — steps in current phase
    phase_time: torch.Tensor = field(init=False)       # (N,) float — seconds in current phase
    station_index: torch.Tensor = field(init=False)    # (N,) int
    stroke_index: torch.Tensor = field(init=False)     # (N,) int — strokes completed this station
    path_progress: torch.Tensor = field(init=False)    # (N,) float — path distance on current stroke
    stroke_direction: torch.Tensor = field(init=False) # (N,) +1 or -1
    completion_hold: torch.Tensor = field(init=False)  # (N,) int — consecutive good steps
    total_strokes: torch.Tensor = field(init=False)    # (N,) int — total strokes across all stations
    accumulated_path: torch.Tensor = field(init=False) # (N,) float — accumulated cutter path

    def __post_init__(self):
        N = self.num_envs
        device = self.device
        self.phase = torch.full((N,), int(StopMowPhase.RESET_SETTLE), device=device, dtype=torch.long)
        self.phase_step = torch.zeros(N, device=device, dtype=torch.long)
        self.phase_time = torch.zeros(N, device=device)
        self.station_index = torch.zeros(N, device=device, dtype=torch.long)
        self.stroke_index = torch.zeros(N, device=device, dtype=torch.long)
        self.path_progress = torch.zeros(N, device=device)
        self.stroke_direction = torch.ones(N, device=device)
        self.completion_hold = torch.zeros(N, device=device, dtype=torch.long)
        self.total_strokes = torch.zeros(N, device=device, dtype=torch.long)
        self.accumulated_path = torch.zeros(N, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset state to initial phase for selected environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(self.device, dtype=torch.long)
        self.phase[env_ids] = int(StopMowPhase.RESET_SETTLE)
        self.phase_step[env_ids] = 0
        self.phase_time[env_ids] = 0.0
        self.station_index[env_ids] = 0
        self.stroke_index[env_ids] = 0
        self.path_progress[env_ids] = 0.0
        self.stroke_direction[env_ids] = 1.0
        self.completion_hold[env_ids] = 0
        self.total_strokes[env_ids] = 0
        self.accumulated_path[env_ids] = 0.0

    def to_phase(self, phase: StopMowPhase, env_ids: torch.Tensor) -> None:
        """Transition selected environments to *phase*, resetting per-phase counters."""
        env_ids = env_ids.to(self.device, dtype=torch.long)
        self.phase[env_ids] = int(phase)
        self.phase_step[env_ids] = 0
        self.phase_time[env_ids] = 0.0
