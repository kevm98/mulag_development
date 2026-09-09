"""Runtime bridge for the cylinder-only command-conditioned PPO2 ROM."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROM_ROOT = _REPO_ROOT / "learned_models/rom_model"  # ROM_pipeline model defs + checkpoints live here
if str(_ROM_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROM_ROOT))

from ROM_pipeline.models import load_checkpoint  # noqa: E402


def pack_desired_cylinder_command(
    target_delta_extension: torch.Tensor,
    target_delta_velocity: torch.Tensor,
) -> torch.Tensor:
    """Pack the command-ROM input, never a valve command.

    The command ROM is a motion-command actuator.  Its external command is
    ``[target_delta_extension(4), target_delta_velocity(4)]``; the ROM predicts
    the valve internally.  Keeping this conversion in one checked function
    prevents the 4-D ID valve interface from being reused by the ROM path.
    """
    if target_delta_extension.ndim != 2 or target_delta_extension.shape[-1] != 4:
        raise ValueError(
            "target_delta_extension must have shape [batch, 4], got "
            f"{tuple(target_delta_extension.shape)}"
        )
    if target_delta_velocity.shape != target_delta_extension.shape:
        raise ValueError(
            "target_delta_velocity must match target_delta_extension shape, got "
            f"{tuple(target_delta_velocity.shape)} vs {tuple(target_delta_extension.shape)}"
        )
    command = torch.cat((target_delta_extension, target_delta_velocity), dim=-1)
    if not torch.isfinite(command).all():
        raise FloatingPointError("Desired cylinder command contains NaN/Inf.")
    return command


@dataclass(frozen=True)
class CommandROMStepOutput:
    predicted_valve: torch.Tensor
    delta_dP: torch.Tensor
    extension_velocity_next: torch.Tensor
    delta_extension: torch.Tensor
    update_mask: torch.Tensor


class CommandCylinderROMRuntime:
    """Execute desired cylinder command -> internal valve -> hydraulics at 50 ms."""

    def __init__(self, *, checkpoint: str | Path, num_envs: int, device, control_dt_s: float) -> None:
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Command ROM checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.model, self.checkpoint = load_checkpoint(self.checkpoint_path, device=self.device)
        self.schema = self.model.schema
        self.sample_time_s = float(self.schema.sample_time_s)
        ratio = self.sample_time_s / float(control_dt_s)
        self.steps_per_tick = int(round(ratio))
        if self.steps_per_tick <= 0 or abs(ratio - self.steps_per_tick) > 1.0e-6:
            raise ValueError(f"ROM dt/control dt must be an integer, got {ratio}.")
        self.history = torch.zeros(
            self.num_envs, self.schema.history_length, self.schema.input_dim, device=self.device
        )
        self.history_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_predicted_valve = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_velocity = torch.zeros(self.num_envs, 4, device=self.device)
        self.delta_extension_per_control_step = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        # Privileged diagnostic only. The deployment controller never consumes
        # this state; the collection pipeline records it to audit the recurrent
        # ROM memory against the learned SD latent.
        self.last_hidden: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids.to(self.device, dtype=torch.long)
        self.history[ids] = 0.0
        self.history_count[ids] = 0
        self.counter[ids] = 0
        self.previous_predicted_valve[ids] = 0.0
        self.last_velocity[ids] = 0.0
        self.delta_extension_per_control_step[ids] = 0.0
        if self.last_hidden is not None:
            self.last_hidden[:, ids] = 0.0

    def _row(self, desired_cylinder_command, extension, velocity, dP) -> torch.Tensor:
        expected4 = (self.num_envs, 4)
        for name, value in (("extension", extension), ("velocity", velocity), ("dP", dP)):
            if value.shape != expected4:
                raise ValueError(f"{name} must have shape {expected4}, got {tuple(value.shape)}")
        if desired_cylinder_command.shape != (self.num_envs, 8):
            raise ValueError(
                "command-ROM command must have shape "
                f"{(self.num_envs, 8)} [delta_extension(4), delta_velocity(4)], "
                f"got {tuple(desired_cylinder_command.shape)}"
            )
        row = torch.cat(
            (extension, velocity, dP, self.previous_predicted_valve, desired_cylinder_command),
            dim=-1,
        )
        if not torch.isfinite(row).all():
            raise FloatingPointError("Command ROM input contains NaN/Inf.")
        return row.to(self.device, dtype=torch.float32)

    def _prospective_history(self, row: torch.Tensor) -> torch.Tensor:
        history = self.history.clone()
        unprimed = self.history_count == 0
        if unprimed.any():
            history[unprimed] = row[unprimed].unsqueeze(1).expand(-1, self.schema.history_length, -1)
        primed = ~unprimed
        if primed.any():
            history[primed, :-1] = history[primed, 1:].clone()
            history[primed, -1] = row[primed]
        return history

    @torch.no_grad()
    def predict_delta(
        self, *, desired_cylinder_command, extension, extension_velocity, dP,
        system_factors=None, rom_b_factors=None,
    ) -> torch.Tensor:
        row = self._row(desired_cylinder_command, extension, extension_velocity, dP)
        return self.model(
            self._prospective_history(row), desired_cylinder_command,
            system_factors=system_factors, rom_b_factors=rom_b_factors,
        ).delta_extension

    @torch.no_grad()
    def step(
        self, *, desired_cylinder_command, extension, extension_velocity, dP,
        system_factors=None, rom_b_factors=None,
    ) -> CommandROMStepOutput:
        row = self._row(desired_cylinder_command, extension, extension_velocity, dP)
        update_mask = self.counter == 0
        ids = torch.where(update_mask)[0]
        predicted_valve = self.previous_predicted_valve.clone()
        delta_dP = torch.zeros(self.num_envs, 4, device=self.device)
        # A model output represents one complete ROM interval (50 ms).  Hold a
        # per-control-step plan so PhysX receives three smooth 60 Hz increments
        # instead of one large 20 Hz target jump followed by two zero updates.
        delta_extension = self.delta_extension_per_control_step.clone()
        velocity_next = self.last_velocity.clone()
        if ids.numel():
            prospective = self._prospective_history(row)
            self.history[ids] = prospective[ids]
            self.history_count[ids] = self.schema.history_length
            selected_factors = None
            selected_rom_b_factors = None
            if system_factors is not None:
                if system_factors.shape != (self.num_envs, 7):
                    raise ValueError(
                        "system_factors must have shape "
                        f"{(self.num_envs, 7)}, got {tuple(system_factors.shape)}"
                    )
                selected_factors = system_factors[ids]
            if rom_b_factors is not None:
                if rom_b_factors.shape != (self.num_envs, 5):
                    raise ValueError(
                        "rom_b_factors must have shape "
                        f"{(self.num_envs, 5)}, got {tuple(rom_b_factors.shape)}"
                    )
                selected_rom_b_factors = rom_b_factors[ids]
            output = self.model(
                self.history[ids], desired_cylinder_command[ids],
                system_factors=selected_factors,
                rom_b_factors=selected_rom_b_factors,
            )
            if self.last_hidden is None:
                self.last_hidden = torch.zeros(
                    output.hidden.shape[0], self.num_envs, output.hidden.shape[-1],
                    device=output.hidden.device, dtype=output.hidden.dtype,
                )
            self.last_hidden[:, ids] = output.hidden.detach()
            values = (output.predicted_valve, output.delta_pressure, output.next_velocity, output.delta_extension)
            if not all(torch.isfinite(value).all() for value in values):
                raise FloatingPointError("Command ROM output contains NaN/Inf.")
            predicted_valve[ids] = output.predicted_valve
            delta_dP[ids] = output.delta_pressure
            velocity_next[ids] = output.next_velocity
            self.delta_extension_per_control_step[ids] = (
                output.delta_extension / float(self.steps_per_tick)
            )
            delta_extension[ids] = self.delta_extension_per_control_step[ids]
        self.previous_predicted_valve = predicted_valve.detach().clone()
        self.last_velocity = velocity_next.detach().clone()
        self.counter[:] = (self.counter + 1) % self.steps_per_tick
        return CommandROMStepOutput(
            predicted_valve, delta_dP, velocity_next, delta_extension, update_mask
        )
