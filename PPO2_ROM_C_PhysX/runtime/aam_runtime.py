"""Online AAM estimator and factor-conditioned command-ROM runtime.

The hidden factors perturb the simulated plant.  Only estimates reconstructed
from causal valve/cylinder history are returned to the PPO policy.
"""

from __future__ import annotations

from pathlib import Path

import torch

from AAM_SystemFactor.models.aam_lstm import AAMLSTM
from PPO2_ROM_B_PhysX.runtime import (
    FACTOR_NAMES as ROM_B_FACTOR_NAMES,
    ROMBCommandCylinderROMRuntime,
    ROMBFactorConfig,
)


BUNDLE_FACTOR_NAMES = (
    "e_comp", "e_flow", "e_pressure",
    "e_seal_1", "e_seal_2", "e_seal_3", "e_seal_4",
)
BUNDLE_TO_ROM_B = (0, 3, 4, 5, 6)
FACTOR_NAMES = ROM_B_FACTOR_NAMES
FACTOR_SCALES = (0.125, 1.0, 1.0, 1.0, 1.0)


class OnlineAAMRuntime:
    """Torch-only, vectorized, causal inference for the trained AAM bundle."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        num_envs: int,
        device,
        control_dt_s: float,
        sample_time_s: float = 0.05,
        ema_beta: float = 0.90,
    ) -> None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"AAM checkpoint not found: {path}")
        self.checkpoint_path = path
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.sample_time_s = float(sample_time_s)
        ratio = self.sample_time_s / float(control_dt_s)
        self.steps_per_tick = int(round(ratio))
        if self.steps_per_tick <= 0 or abs(ratio - self.steps_per_tick) > 1.0e-6:
            raise ValueError(f"AAM dt/control dt must be an integer, got {ratio}")
        self.ema_beta = min(max(float(ema_beta), 0.0), 0.9999)

        bundle = torch.load(path, map_location=self.device, weights_only=False)
        expected_names = list(BUNDLE_FACTOR_NAMES)
        if list(bundle.get("target_names", ())) != expected_names:
            raise ValueError("AAM target order does not match PPO2_C factor contract")
        self.history_length = int(bundle.get("T", 10))
        if int(bundle["input_size"]) != 20 or int(bundle["output_size"]) != 7:
            raise ValueError("PPO2_C requires an AAM with 20 inputs and 7 outputs")
        self.model = AAMLSTM(
            input_size=int(bundle["input_size"]),
            hidden_size=int(bundle["hidden_size"]),
            num_layers=int(bundle["num_layers"]),
            output_size=int(bundle["output_size"]),
        ).to(self.device)
        self.model.load_state_dict(bundle["model_state_dict"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.input_mean = torch.as_tensor(
            bundle["input_scaler_mean"], dtype=torch.float32, device=self.device
        ).reshape(1, 1, 20)
        self.input_std = torch.as_tensor(
            bundle["input_scaler_std"], dtype=torch.float32, device=self.device
        ).reshape(1, 1, 20).clamp_min(1.0e-8)
        self.target_mean = torch.as_tensor(
            bundle["target_scaler_mean"], dtype=torch.float32, device=self.device
        ).reshape(1, 7)
        self.target_std = torch.as_tensor(
            bundle["target_scaler_std"], dtype=torch.float32, device=self.device
        ).reshape(1, 7).clamp_min(1.0e-8)
        self.factor_scales = torch.as_tensor(
            FACTOR_SCALES, dtype=torch.float32, device=self.device
        ).reshape(1, len(FACTOR_NAMES))
        self.history = torch.zeros(
            self.num_envs, self.history_length, 20,
            dtype=torch.float32, device=self.device,
        )
        self.history_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.counter = torch.zeros_like(self.history_count)
        self.physical_factors = torch.zeros(
            self.num_envs, len(FACTOR_NAMES), dtype=torch.float32, device=self.device
        )
        self.policy_factors = torch.zeros_like(self.physical_factors)
        self.ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids.to(self.device, dtype=torch.long)
        self.history[ids] = 0.0
        self.history_count[ids] = 0
        self.counter[ids] = 0
        self.physical_factors[ids] = 0.0
        self.policy_factors[ids] = 0.0
        self.ready[ids] = False

    @torch.no_grad()
    def observe(
        self, *, valve: torch.Tensor, extension: torch.Tensor,
        velocity: torch.Tensor, dP: torch.Tensor, force: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = (valve, extension, velocity, dP, force)
        if any(tuple(value.shape) != (self.num_envs, 4) for value in values):
            raise ValueError("each AAM signal must have shape [num_envs, 4]")
        row = torch.cat(values, dim=-1).to(self.device, dtype=torch.float32)
        if not torch.isfinite(row).all():
            raise FloatingPointError("AAM input contains NaN/Inf")

        update = self.counter == 0
        ids = torch.where(update)[0]
        if ids.numel():
            self.history[ids, :-1] = self.history[ids, 1:].clone()
            self.history[ids, -1] = row[ids]
            self.history_count[ids] = torch.minimum(
                self.history_count[ids] + 1,
                torch.full_like(self.history_count[ids], self.history_length),
            )
            infer_ids = ids[self.history_count[ids] >= self.history_length]
            if infer_ids.numel():
                normalized = (
                    self.history[infer_ids] - self.input_mean
                ) / self.input_std
                predicted = self.model(normalized)
                predicted = predicted * self.target_std + self.target_mean
                # The current AAM training target accidentally encoded seals as
                # 1 + w_seal/0.25.  Removing that constant restores the stated
                # physical contract without changing estimation residuals.
                predicted[:, 3:] -= 1.0
                # ROM-C consumes only the five factors that perturb ROM-B.
                # Flow and relief-pressure outputs belong to the older
                # Simscape bundle and are deliberately discarded.
                predicted = predicted[:, BUNDLE_TO_ROM_B]
                predicted = torch.nan_to_num(predicted).clamp(
                    min=torch.tensor(
                        [-0.125, 0.0, 0.0, 0.0, 0.0], device=self.device
                    ),
                    max=torch.tensor(
                        [0.02, 1.0, 1.0, 1.0, 1.0], device=self.device
                    ),
                )
                first = ~self.ready[infer_ids]
                smoothed = (
                    self.ema_beta * self.physical_factors[infer_ids]
                    + (1.0 - self.ema_beta) * predicted
                )
                smoothed[first] = predicted[first]
                self.physical_factors[infer_ids] = smoothed
                self.policy_factors[infer_ids] = torch.clamp(
                    smoothed / self.factor_scales, -1.0, 1.0
                )
                self.ready[infer_ids] = True
        self.counter[:] = (self.counter + 1) % self.steps_per_tick
        return (
            self.physical_factors.detach().clone(),
            self.policy_factors.detach().clone(),
            self.ready.detach().clone(),
        )


class AdaptiveCommandCylinderROMRuntime(ROMBCommandCylinderROMRuntime):
    """Compatibility name for the exact five-factor ROM-B plant runtime."""

    def __init__(
        self,
        *,
        randomize_factors: bool = True,
        factor_low=(-0.125, 0.0, 0.0, 0.0, 0.0),
        factor_high=(0.02, 1.0, 1.0, 1.0, 1.0),
        **kwargs,
    ) -> None:
        if tuple(factor_low) != (-0.125, 0.0, 0.0, 0.0, 0.0) or tuple(
            factor_high
        ) != (0.02, 1.0, 1.0, 1.0, 1.0):
            raise ValueError("ROM-C must use the audited ROM-B factor ranges")
        super().__init__(
            **kwargs,
            factor_cfg=ROMBFactorConfig(
                mode="random" if randomize_factors else "nominal",
                randomization_scale=1.0,
            ),
        )

    @property
    def true_system_factors(self) -> torch.Tensor:
        """Legacy read-only alias; the policy never receives this tensor."""

        return self.true_factors
