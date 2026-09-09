"""Adapter from the trained base hydraulic ROM to the ID/FD control runtime."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROM_PROJECT_ROOT = _REPO_ROOT / "learned_models/rom_model"  # ROM_pipeline model defs + checkpoints live here
if str(_ROM_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROM_PROJECT_ROOT))

from ROM_pipeline.models import load_base_checkpoint  # noqa: E402


@dataclass(frozen=True)
class BaseCylinderROMStepOutput:
    """One control-step view of the native-50 ms ROM prediction."""

    delta_extension: torch.Tensor
    extension_velocity_next: torch.Tensor
    delta_dP: torch.Tensor
    predicted_force: torch.Tensor
    update_mask: torch.Tensor


class BaseCylinderROMRuntime:
    """Run the trained 50 ms base ROM inside a faster vectorized control loop.

    The full ROM displacement is emitted once on each ROM tick and zero is
    emitted on intervening control steps. The outer position-target integrator
    therefore applies the model displacement exactly once rather than dividing
    or repeatedly accumulating it.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        num_envs: int,
        device: str | torch.device,
        control_dt_s: float,
    ) -> None:
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Base ROM checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        self.num_envs = int(num_envs)
        self.model, self.checkpoint = load_base_checkpoint(self.checkpoint_path, device=self.device)
        self.model.eval()
        self.schema = self.model.schema
        self.sample_time_s = float(self.schema.sample_time_s)
        self.control_dt_s = float(control_dt_s)
        ratio = self.sample_time_s / self.control_dt_s
        self.steps_per_rom_tick = int(round(ratio))
        if self.steps_per_rom_tick <= 0 or abs(ratio - self.steps_per_rom_tick) > 1.0e-6:
            raise ValueError(
                "Base ROM sample time must be an integer multiple of control_dt: "
                f"rom_dt={self.sample_time_s}, control_dt={self.control_dt_s}, ratio={ratio}."
            )

        self.history = torch.zeros(
            (self.num_envs, self.schema.history_length, self.schema.input_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.history_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.tick_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_extension_velocity = torch.zeros((self.num_envs, 4), device=self.device)
        self.last_predicted_force = torch.zeros((self.num_envs, 4), device=self.device)
        self.last_update_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            ids = slice(None)
        else:
            ids = env_ids.to(device=self.device, dtype=torch.long)
        self.history[ids] = 0.0
        self.history_count[ids] = 0
        self.tick_counter[ids] = 0
        self.last_extension_velocity[ids] = 0.0
        self.last_predicted_force[ids] = 0.0
        self.last_update_mask[ids] = False

    @staticmethod
    def _require_shape(name: str, value: torch.Tensor, shape: tuple[int, int]) -> None:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")

    def _input_row(
        self,
        *,
        valve_cmd: torch.Tensor,
        extension: torch.Tensor,
        extension_velocity: torch.Tensor,
        dP: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        batch = (self.num_envs, 4)
        for name, value in (
            ("valve_cmd", valve_cmd),
            ("extension", extension),
            ("extension_velocity", extension_velocity),
            ("dP", dP),
            ("force", force),
        ):
            self._require_shape(name, value, batch)
        row = torch.cat(
            (valve_cmd, extension, extension_velocity, dP, force), dim=-1
        ).to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(row).all():
            raise FloatingPointError("Base ROM runtime input contains NaN/Inf.")
        return row

    @torch.no_grad()
    def predict_delta(
        self,
        *,
        valve_cmd: torch.Tensor,
        extension: torch.Tensor,
        extension_velocity: torch.Tensor,
        dP: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a native ROM-tick displacement without mutating runtime state."""
        row = self._input_row(
            valve_cmd=valve_cmd,
            extension=extension,
            extension_velocity=extension_velocity,
            dP=dP,
            force=force,
        )
        history = self.history.clone()
        unprimed = self.history_count == 0
        primed = ~unprimed
        if unprimed.any():
            history[unprimed] = row[unprimed].unsqueeze(1).expand(-1, self.schema.history_length, -1)
        if primed.any():
            history[primed, :-1] = history[primed, 1:].clone()
            history[primed, -1] = row[primed]
        output = self.model(history, valve_cmd)
        if not torch.isfinite(output.delta_extension).all():
            raise FloatingPointError("Base ROM prediction contains NaN/Inf.")
        return output.delta_extension

    @torch.no_grad()
    def step(
        self,
        *,
        valve_cmd: torch.Tensor,
        extension: torch.Tensor,
        extension_velocity: torch.Tensor,
        dP: torch.Tensor,
        force: torch.Tensor,
    ) -> BaseCylinderROMStepOutput:
        row = self._input_row(
            valve_cmd=valve_cmd,
            extension=extension,
            extension_velocity=extension_velocity,
            dP=dP,
            force=force,
        )

        update_mask = self.tick_counter == 0
        update_ids = torch.where(update_mask)[0]
        delta_extension = torch.zeros((self.num_envs, 4), device=self.device)
        delta_dP = torch.zeros_like(delta_extension)
        velocity_next = self.last_extension_velocity.clone()
        predicted_force = self.last_predicted_force.clone()

        if update_ids.numel():
            unprimed = update_ids[self.history_count[update_ids] == 0]
            primed = update_ids[self.history_count[update_ids] > 0]
            if unprimed.numel():
                self.history[unprimed] = row[unprimed].unsqueeze(1).expand(
                    -1, self.schema.history_length, -1
                )
                self.history_count[unprimed] = self.schema.history_length
            if primed.numel():
                self.history[primed, :-1] = self.history[primed, 1:].clone()
                self.history[primed, -1] = row[primed]
                self.history_count[primed].clamp_max_(self.schema.history_length - 1).add_(1)

            output = self.model(self.history[update_ids], valve_cmd[update_ids])
            values = (
                output.delta_extension,
                output.delta_pressure,
                output.next_velocity,
                output.predicted_force,
            )
            if not all(torch.isfinite(value).all() for value in values):
                raise FloatingPointError("Base ROM runtime output contains NaN/Inf.")
            delta_extension[update_ids] = output.delta_extension
            delta_dP[update_ids] = output.delta_pressure
            velocity_next[update_ids] = output.next_velocity
            predicted_force[update_ids] = output.predicted_force

        self.tick_counter[:] = (self.tick_counter + 1) % self.steps_per_rom_tick
        self.last_extension_velocity = velocity_next.detach().clone()
        self.last_predicted_force = predicted_force.detach().clone()
        self.last_update_mask = update_mask.detach().clone()
        return BaseCylinderROMStepOutput(
            delta_extension=delta_extension,
            extension_velocity_next=velocity_next,
            delta_dP=delta_dP,
            predicted_force=predicted_force,
            update_mask=update_mask,
        )
