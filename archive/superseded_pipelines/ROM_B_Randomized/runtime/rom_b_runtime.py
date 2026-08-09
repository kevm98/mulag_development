"""ROM-B wrapper around the fixed PPO2 command-conditioned cylinder ROM."""

from __future__ import annotations

from pathlib import Path

import torch

from PPO2_ROM_PhysX.runtime.command_rom_runtime import (
    CommandCylinderROMRuntime,
    CommandROMStepOutput,
)

from .curriculum import ROMBCurriculumController, ROMBCurriculumThresholds
from .randomization import ROMBRandomizationConfig, ROMBRandomizer


class ROMBCommandCylinderROMRuntime:
    """Hydraulically randomized family around one fixed reference ROM.

    The reference model is never retrained and is called without its
    ``system_factors`` hook.  ROM-B applies residual plant variation outside the
    model, restricted to command-side flow/valve timing and pressure-channel
    corrections.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        num_envs: int | None = None,
        device: str | torch.device | None = None,
        control_dt_s: float | None = None,
        randomization_cfg: ROMBRandomizationConfig | None = None,
        curriculum: ROMBCurriculumController | None = None,
        reference_runtime: CommandCylinderROMRuntime | None = None,
    ) -> None:
        if reference_runtime is None:
            if checkpoint is None or num_envs is None or device is None or control_dt_s is None:
                raise ValueError(
                    "checkpoint, num_envs, device and control_dt_s are required when "
                    "reference_runtime is not provided."
                )
            reference_runtime = CommandCylinderROMRuntime(
                checkpoint=checkpoint,
                num_envs=int(num_envs),
                device=device,
                control_dt_s=float(control_dt_s),
            )
        self.reference = reference_runtime
        self.randomizer = ROMBRandomizer(
            num_envs=self.reference.num_envs,
            device=self.reference.device,
            config=randomization_cfg,
        )
        self.curriculum = curriculum or ROMBCurriculumController(
            stage=self.randomizer.config.curriculum_stage,
            enabled=False,
            thresholds=ROMBCurriculumThresholds(),
        )
        self.last_reference_output: CommandROMStepOutput | None = None
        self.last_rom_command = torch.zeros(self.reference.num_envs, 8, device=self.reference.device)

    @property
    def checkpoint_path(self):
        return self.reference.checkpoint_path

    @property
    def device(self):
        return self.reference.device

    @property
    def num_envs(self):
        return self.reference.num_envs

    @property
    def model(self):
        return self.reference.model

    @property
    def checkpoint(self):
        return self.reference.checkpoint

    @property
    def schema(self):
        return self.reference.schema

    @property
    def sample_time_s(self):
        return self.reference.sample_time_s

    @property
    def steps_per_tick(self):
        return self.reference.steps_per_tick

    @property
    def counter(self):
        return self.reference.counter

    @property
    def history(self):
        return self.reference.history

    @property
    def history_count(self):
        return self.reference.history_count

    @property
    def previous_predicted_valve(self):
        return self.reference.previous_predicted_valve

    @property
    def last_velocity(self):
        return self.reference.last_velocity

    @property
    def delta_extension_per_control_step(self):
        return self.reference.delta_extension_per_control_step

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.reference.reset(env_ids)
        self.randomizer.reset(env_ids)
        if env_ids is None:
            self.last_rom_command.zero_()
        else:
            self.last_rom_command[env_ids.to(device=self.device, dtype=torch.long)] = 0.0

    def set_fixed_factors(
        self,
        *,
        env_ids: torch.Tensor | None = None,
        beta_eff_pa: float = 0.7e9,
        flow_scale: float = 1.0,
        pressure_scale: float = 1.0,
        seal_factor: torch.Tensor | float = 0.0,
    ) -> None:
        self.randomizer.set_fixed(
            env_ids=env_ids,
            beta_eff_pa=beta_eff_pa,
            flow_scale=flow_scale,
            pressure_scale=pressure_scale,
            seal_factor=seal_factor,
        )

    @torch.no_grad()
    def predict_delta(
        self,
        *,
        command,
        extension,
        extension_velocity,
        dP,
        system_factors=None,
    ) -> torch.Tensor:
        if system_factors is not None:
            raise ValueError("ROM-B does not accept AAM/system_factors; factors are hidden from PPO.")
        update_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        rom_command = self.randomizer.transform_command(
            command,
            update_mask=update_mask,
            dt_model=self.sample_time_s,
        )
        return self.reference.predict_delta(
            command=rom_command,
            extension=extension,
            extension_velocity=extension_velocity,
            dP=dP,
            system_factors=None,
        )

    @torch.no_grad()
    def step(
        self,
        *,
        command,
        extension,
        extension_velocity,
        dP,
        system_factors=None,
    ) -> CommandROMStepOutput:
        if system_factors is not None:
            raise ValueError("ROM-B does not accept AAM/system_factors; factors are hidden from PPO.")
        update_mask_before = self.reference.counter == 0
        rom_command = self.randomizer.transform_command(
            command,
            update_mask=update_mask_before,
            dt_model=self.sample_time_s,
        )
        self.last_rom_command = rom_command.detach().clone()
        ref_out = self.reference.step(
            command=rom_command,
            extension=extension,
            extension_velocity=extension_velocity,
            dP=dP,
            system_factors=None,
        )
        delta_dP = self.randomizer.apply_pressure_randomization(
            dP_current=dP,
            delta_dP_ref=ref_out.delta_dP,
            update_mask=ref_out.update_mask,
            dt_model=self.sample_time_s,
        )
        self.randomizer.update_effective_valve(ref_out.predicted_valve, ref_out.update_mask)
        out = CommandROMStepOutput(
            predicted_valve=ref_out.predicted_valve,
            delta_dP=delta_dP,
            extension_velocity_next=ref_out.extension_velocity_next,
            delta_extension=ref_out.delta_extension,
            update_mask=ref_out.update_mask,
        )
        self.last_reference_output = ref_out
        return out

    def factor_log_scalars(self) -> dict[str, torch.Tensor]:
        return self.randomizer.factor_log_scalars()

    def maybe_update_curriculum(self, metrics: dict[str, float]) -> int:
        stage = self.curriculum.update(metrics)
        if stage != self.randomizer.config.curriculum_stage:
            cfg = self.randomizer.config
            self.randomizer.config = ROMBRandomizationConfig(
                mode=cfg.mode,
                curriculum_stage=stage,
                beta_ref_pa=cfg.beta_ref_pa,
                pressure_ref_pa=cfg.pressure_ref_pa,
                nominal_valve_tau_s=cfg.nominal_valve_tau_s,
                nominal_leakage_decay_per_s=cfg.nominal_leakage_decay_per_s,
                command_abs_limit=cfg.command_abs_limit,
                enable_pressure_saturation=cfg.enable_pressure_saturation,
                preserve_nominal_equivalence=cfg.preserve_nominal_equivalence,
            )
        return stage
