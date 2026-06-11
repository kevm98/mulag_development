from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


Predictor = Callable[[torch.Tensor], torch.Tensor | dict[str, torch.Tensor]]


@dataclass(frozen=True)
class MultirateROMConfig:
    control_dt: float = 0.01
    rom_dt: float = 0.05
    input_dim: int = 26
    output_dim: int = 4
    history_length: int = 10

    @property
    def ratio(self) -> int:
        raw = self.rom_dt / self.control_dt
        rounded = int(round(raw))
        if abs(raw - rounded) > 1.0e-9:
            raise ValueError(f"rom_dt/control_dt must be an integer ratio, got {raw}.")
        if rounded <= 0:
            raise ValueError("rom_dt/control_dt ratio must be positive.")
        return rounded


class MultirateROMWrapper:
    """Offline 0.01 s control-step wrapper around a 0.05 s cylinder ROM.

    The wrapped predictor receives raw ROM histories sampled at 0.05 s spacing:
    `[num_envs, history_length, 26] -> [num_envs, 4]`.
    """

    def __init__(
        self,
        predictor: Predictor,
        *,
        num_envs: int,
        config: MultirateROMConfig = MultirateROMConfig(),
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.predictor = predictor
        self.config = config
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.substeps_per_rom_step = config.ratio
        self.history = torch.zeros(
            (self.num_envs, config.history_length, config.input_dim),
            dtype=dtype,
            device=self.device,
        )
        self.history_filled = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.rom_step_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.predicted_delta_extension_50ms = torch.zeros(
            (self.num_envs, config.output_dim),
            dtype=dtype,
            device=self.device,
        )
        self.applied_delta_extension_10ms = torch.zeros_like(self.predicted_delta_extension_50ms)
        self.predicted_delta_extension_kinematic_50ms = torch.zeros_like(self.predicted_delta_extension_50ms)
        self.predicted_delta_extension_correction_50ms = torch.zeros_like(self.predicted_delta_extension_50ms)
        self.rom_update_flag = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.history_ready = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | list[int] | tuple[int, ...] | None = None) -> None:
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.history[env_ids_t] = 0.0
        self.history_filled[env_ids_t] = 0
        self.rom_step_counter[env_ids_t] = 0
        self.predicted_delta_extension_50ms[env_ids_t] = 0.0
        self.applied_delta_extension_10ms[env_ids_t] = 0.0
        self.predicted_delta_extension_kinematic_50ms[env_ids_t] = 0.0
        self.predicted_delta_extension_correction_50ms[env_ids_t] = 0.0
        self.rom_update_flag[env_ids_t] = False
        self.history_ready[env_ids_t] = False

    def prime_history(self, history: torch.Tensor, env_ids: torch.Tensor | list[int] | tuple[int, ...] | None = None) -> None:
        history = history.to(device=self.device, dtype=self.dtype)
        if env_ids is None:
            if history.shape != self.history.shape:
                raise ValueError(f"history must have shape {tuple(self.history.shape)}, got {tuple(history.shape)}.")
            env_ids_t = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids_t = env_ids.to(device=self.device, dtype=torch.long)
            else:
                env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            expected = (int(env_ids_t.numel()), self.config.history_length, self.config.input_dim)
            if history.shape != expected:
                raise ValueError(f"history must have shape {expected}, got {tuple(history.shape)}.")
        if not torch.isfinite(history).all():
            raise ValueError("history contains NaN/Inf.")
        self.history[env_ids_t] = history
        self.history_filled[env_ids_t] = self.config.history_length
        self.history_ready[env_ids_t] = True

    def _validate_observation(self, observation_10ms: torch.Tensor) -> torch.Tensor:
        observation = observation_10ms.to(device=self.device, dtype=self.dtype)
        expected = (self.num_envs, self.config.input_dim)
        if observation.shape != expected:
            raise ValueError(f"observation_10ms must have shape {expected}, got {tuple(observation.shape)}.")
        if not torch.isfinite(observation).all():
            raise ValueError("observation_10ms contains NaN/Inf.")
        return observation

    def _insert_history_sample(self, env_mask: torch.Tensor, observation: torch.Tensor) -> None:
        if not bool(env_mask.any()):
            return
        env_ids = torch.nonzero(env_mask, as_tuple=False).reshape(-1)
        self.history[env_ids, :-1, :] = self.history[env_ids, 1:, :].clone()
        self.history[env_ids, -1, :] = observation[env_ids]
        self.history_filled[env_ids] = torch.clamp(self.history_filled[env_ids] + 1, max=self.config.history_length)
        self.history_ready[env_ids] = self.history_filled[env_ids] >= self.config.history_length

    def _predict(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            raw = self.predictor(history)
        extras: dict[str, torch.Tensor] = {}
        if isinstance(raw, dict):
            if "final_delta_extension" in raw:
                delta = raw["final_delta_extension"]
            elif "delta_extension" in raw:
                delta = raw["delta_extension"]
            else:
                raise KeyError("Predictor dict must contain final_delta_extension or delta_extension.")
            for name in ("delta_extension_kinematic", "delta_extension_correction"):
                if name in raw:
                    extras[name] = raw[name].to(device=self.device, dtype=self.dtype)
        else:
            delta = raw
        delta = delta.to(device=self.device, dtype=self.dtype)
        expected = (history.shape[0], self.config.output_dim)
        if delta.shape != expected:
            raise ValueError(f"Predictor must return shape {expected}, got {tuple(delta.shape)}.")
        if self.config.output_dim != 4:
            raise ValueError("First deployment contract requires exactly four ROM actuator outputs.")
        if not torch.isfinite(delta).all():
            raise ValueError("Predictor returned NaN/Inf.")
        for name, value in extras.items():
            if value.shape != expected:
                raise ValueError(f"Predictor {name} must return shape {expected}, got {tuple(value.shape)}.")
            if not torch.isfinite(value).all():
                raise ValueError(f"Predictor {name} returned NaN/Inf.")
        return {"final_delta_extension": delta, **extras}

    def step(self, observation_10ms: torch.Tensor) -> torch.Tensor:
        observation = self._validate_observation(observation_10ms)
        self.rom_update_flag.zero_()
        update_mask = self.rom_step_counter == 0
        self._insert_history_sample(update_mask, observation)

        ready_update = update_mask & self.history_ready
        if bool(ready_update.any()):
            env_ids = torch.nonzero(ready_update, as_tuple=False).reshape(-1)
            prediction = self._predict(self.history[env_ids])
            predicted = prediction["final_delta_extension"]
            self.predicted_delta_extension_50ms[env_ids] = predicted
            self.applied_delta_extension_10ms[env_ids] = predicted / float(self.substeps_per_rom_step)
            if "delta_extension_kinematic" in prediction:
                self.predicted_delta_extension_kinematic_50ms[env_ids] = prediction["delta_extension_kinematic"]
            else:
                self.predicted_delta_extension_kinematic_50ms[env_ids] = 0.0
            if "delta_extension_correction" in prediction:
                self.predicted_delta_extension_correction_50ms[env_ids] = prediction["delta_extension_correction"]
            else:
                self.predicted_delta_extension_correction_50ms[env_ids] = 0.0
            self.rom_update_flag[env_ids] = True

        self.rom_step_counter = (self.rom_step_counter + 1) % self.substeps_per_rom_step
        if not torch.isfinite(self.applied_delta_extension_10ms).all():
            raise ValueError("applied_delta_extension_10ms contains NaN/Inf.")
        return self.applied_delta_extension_10ms.clone()

    def debug_state(self) -> dict[str, Any]:
        return {
            "rom_update_flag": self.rom_update_flag.clone(),
            "rom_step_counter": self.rom_step_counter.clone(),
            "predicted_delta_extension_50ms": self.predicted_delta_extension_50ms.clone(),
            "applied_delta_extension_10ms": self.applied_delta_extension_10ms.clone(),
            "predicted_delta_extension_kinematic_50ms": self.predicted_delta_extension_kinematic_50ms.clone(),
            "predicted_delta_extension_correction_50ms": self.predicted_delta_extension_correction_50ms.clone(),
            "history_ready": self.history_ready.clone(),
            "substeps_per_rom_step": self.substeps_per_rom_step,
        }
