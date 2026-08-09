from __future__ import annotations

from dataclasses import dataclass

import torch

from .base_rom_actuator import BaseROMActuator, BaseRuntimeOutput


@dataclass(frozen=True)
class BaseMultirateConfig:
    control_dt_s: float = 0.01
    rom_dt_s: float = 0.05

    @property
    def steps_per_rom_tick(self) -> int:
        ratio = self.rom_dt_s / self.control_dt_s
        rounded = int(round(ratio))
        if rounded <= 0 or abs(ratio - rounded) > 1.0e-9:
            raise ValueError(f"rom_dt/control_dt must be a positive integer, got {ratio}.")
        return rounded


class BaseROMMultirate:
    """Run the base ROM at 0.05 s and hold its absolute target between ticks.

    A predicted 0.05 s displacement is applied once at the ROM tick. It is not
    divided by the number of physics/control steps.
    """

    def __init__(
        self,
        actuator: BaseROMActuator,
        *,
        num_envs: int,
        config: BaseMultirateConfig = BaseMultirateConfig(),
    ) -> None:
        if abs(config.rom_dt_s - actuator.schema.sample_time_s) > 1.0e-9:
            raise ValueError("Multirate ROM timestep does not match the checkpoint schema.")
        self.actuator = actuator
        self.config = config
        self.counter = torch.zeros(num_envs, dtype=torch.long, device=actuator.model.input_mean.device)
        self.last_output: BaseRuntimeOutput | None = None

    def reset(self, measured_extension: torch.Tensor, current_row: torch.Tensor | None = None) -> None:
        self.actuator.reset(measured_extension, current_row)
        self.counter.zero_()
        self.last_output = None

    @torch.no_grad()
    def step(
        self,
        *,
        state_row: torch.Tensor,
        valve_command: torch.Tensor,
        measured_extension: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, BaseRuntimeOutput | None]:
        # All vectorized environments share the simulation clock.
        update = int(self.counter[0]) == 0
        if update:
            self.last_output = self.actuator.step(
                state_row=state_row, valve_command=valve_command, measured_extension=measured_extension
            )
        self.counter[:] = (self.counter + 1) % self.config.steps_per_rom_tick
        target = self.actuator.targets.target
        if target is None:
            target = measured_extension
        return target.clone(), update, self.last_output
