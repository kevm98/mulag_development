from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import torch

try:
    from models import ModelConfig, build_model
    from rom_data_contract import (
        AUGMENTED_INPUT_FEATURE_NAMES,
        CORE_INPUT_FEATURE_NAMES,
        FORCE_FEATURE_NAMES,
        HISTORY_LENGTH,
        TARGET_FEATURE_NAMES,
    )
except ImportError:
    from kevin_integration.rom_cylinders.models import ModelConfig, build_model
    from kevin_integration.rom_cylinders.rom_data_contract import (
        AUGMENTED_INPUT_FEATURE_NAMES,
        CORE_INPUT_FEATURE_NAMES,
        FORCE_FEATURE_NAMES,
        HISTORY_LENGTH,
        TARGET_FEATURE_NAMES,
    )


ROM_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORT_DIR = ROM_DIR / "exported"


class LearnedCylinderROM:
    """Inference wrapper for an exported hydraulic cylinder ROM."""

    def __init__(
        self,
        export_dir: str | Path = DEFAULT_EXPORT_DIR,
        *,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        extension_delta_clamp: float | None = None,
        dp_delta_clamp: float | None = None,
        strict_finite: bool = True,
    ) -> None:
        self.export_dir = Path(export_dir)
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.extension_delta_clamp = extension_delta_clamp
        self.dp_delta_clamp = dp_delta_clamp
        self.strict_finite = bool(strict_finite)

        model_payload = torch.load(self.export_dir / "best_rom.pth", map_location=self.device)
        scaler_payload = torch.load(self.export_dir / "best_rom_scaler.pth", map_location=self.device)
        self.metadata = json.loads((self.export_dir / "best_rom_metadata.json").read_text(encoding="utf-8"))
        self.config = json.loads((self.export_dir / "best_rom_config.json").read_text(encoding="utf-8"))

        self.use_force_input = bool(self.metadata["use_force_input"])
        self.target_mode = str(self.metadata.get("target_mode", self.config.get("target_mode", "extension_and_dp")))
        expected_inputs = AUGMENTED_INPUT_FEATURE_NAMES if self.use_force_input else CORE_INPUT_FEATURE_NAMES
        expected_targets = TARGET_FEATURE_NAMES[:4] if self.target_mode == "extension_only" else TARGET_FEATURE_NAMES
        if tuple(self.metadata["input_feature_order"]) != expected_inputs:
            raise ValueError("Exported ROM input feature order does not match rom_data_contract.py.")
        if tuple(self.metadata["target_order"]) != expected_targets:
            raise ValueError("Exported ROM target order does not match rom_data_contract.py.")
        if self.use_force_input and tuple(self.metadata["force_feature_order"]) != FORCE_FEATURE_NAMES:
            raise ValueError("Exported ROM Force feature order does not match rom_data_contract.py.")

        self.history_length = int(self.metadata["history_length"])
        if self.history_length != HISTORY_LENGTH:
            raise ValueError(f"Expected history length {HISTORY_LENGTH}, got {self.history_length}.")
        self.input_dim = int(self.metadata["input_dimension"])
        self.target_dim = int(self.metadata["target_dimension"])
        if self.target_dim not in {4, 8}:
            raise ValueError(f"Expected target dimension 4 or 8, got {self.target_dim}.")
        self.input_mean = scaler_payload["input_mean"].to(self.device).float().reshape(1, 1, -1)
        self.input_std = torch.clamp(scaler_payload["input_std"].to(self.device).float().reshape(1, 1, -1), min=1.0e-12)
        self.target_mean = scaler_payload["target_mean"].to(self.device).float().reshape(1, -1)
        self.target_std = torch.clamp(scaler_payload["target_std"].to(self.device).float().reshape(1, -1), min=1.0e-12)

        model_config = ModelConfig(
            model_type=self.config["model_type"],
            input_dim=self.input_dim,
            output_dim=self.target_dim,
            target_mode=str(self.config.get("target_mode", self.target_mode)),
            hidden_dim=int(self.config.get("hidden_dim", 64) or 64),
            num_layers=int(self.config.get("num_layers", 2) or 2),
            dropout=float(self.config.get("dropout", 0.0) or 0.0),
            bounded_outputs=bool(self.config.get("bounded_outputs", False)),
            output_scale=tuple(self.config["output_scale"]) if self.config.get("output_scale") is not None else None,
        )
        normalization_stats = SimpleNamespace(
            input_mean=self.input_mean.reshape(-1).detach().cpu(),
            input_std=self.input_std.reshape(-1).detach().cpu(),
            target_mean=self.target_mean.reshape(-1).detach().cpu(),
            target_std=self.target_std.reshape(-1).detach().cpu(),
            target_mode=self.target_mode,
        )
        self.model = build_model(model_config, normalization_stats=normalization_stats)
        self.model.load_state_dict(model_payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.history = torch.zeros(
            (self.num_envs, self.history_length, self.input_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.history_filled = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | list[int] | tuple[int, ...] | None = None) -> None:
        if env_ids is None:
            self.history.zero_()
            self.history_filled.zero_()
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.history[env_ids] = 0.0
        self.history_filled[env_ids] = 0

    def _as_state(self, value: torch.Tensor, name: str, expected_dim: int) -> torch.Tensor:
        value = value.to(device=self.device, dtype=torch.float32)
        if value.shape != (self.num_envs, expected_dim):
            raise ValueError(f"{name} must have shape ({self.num_envs}, {expected_dim}), got {tuple(value.shape)}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/Inf.")
        return value

    def assemble_features(
        self,
        *,
        valve_cmd: torch.Tensor,
        dP: torch.Tensor,
        extension: torch.Tensor,
        extension_velocity: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valve_cmd = self._as_state(valve_cmd, "valve_cmd", 4)
        dP = self._as_state(dP, "dP", 4)
        extension = self._as_state(extension, "extension", 4)
        extension_velocity = self._as_state(extension_velocity, "extension_velocity", 4)
        joint_pos = self._as_state(joint_pos, "joint_pos", 5)
        joint_vel = self._as_state(joint_vel, "joint_vel", 5)
        parts = [valve_cmd, dP, extension, extension_velocity, joint_pos, joint_vel]
        if self.use_force_input:
            if force is None:
                raise ValueError("This exported ROM requires Force input, but force=None was provided.")
            parts.append(self._as_state(force, "force", 4))
        elif force is not None:
            raise ValueError("This exported ROM is a core model and must not receive Force input.")
        features = torch.cat(parts, dim=-1)
        if features.shape[-1] != self.input_dim:
            raise RuntimeError(f"Assembled feature dimension {features.shape[-1]} != expected {self.input_dim}.")
        return features

    def append_features(self, features: torch.Tensor) -> None:
        features = self._as_state(features, "features", self.input_dim)
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1, :] = features
        self.history_filled = torch.clamp(self.history_filled + 1, max=self.history_length)

    def set_history_from_sequence(self, raw_sequence: torch.Tensor) -> None:
        raw_sequence = raw_sequence.to(device=self.device, dtype=torch.float32)
        if raw_sequence.shape != (self.num_envs, self.history_length, self.input_dim):
            raise ValueError(
                f"raw_sequence must have shape ({self.num_envs}, {self.history_length}, {self.input_dim}), "
                f"got {tuple(raw_sequence.shape)}."
            )
        if not torch.isfinite(raw_sequence).all():
            raise ValueError("raw_sequence contains NaN/Inf.")
        self.history = raw_sequence.clone()
        self.history_filled.fill_(self.history_length)

    def _predict_from_history(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool]]:
        x_norm = (self.history - self.input_mean) / self.input_std
        with torch.no_grad():
            pred_norm = self.model(x_norm)
            pred_raw = pred_norm * self.target_std + self.target_mean
        output_nan_or_inf = ~torch.isfinite(pred_raw).all(dim=-1)
        if bool(output_nan_or_inf.any()) and self.strict_finite:
            raise FloatingPointError("ROM prediction contains NaN/Inf.")
        delta_extension = pred_raw[:, 0:4]
        has_delta_dp = pred_raw.shape[-1] >= 8
        delta_dp = pred_raw[:, 4:8] if has_delta_dp else torch.zeros(
            (self.num_envs, 4),
            dtype=pred_raw.dtype,
            device=pred_raw.device,
        )
        extension_clamped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        dp_clamped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if self.extension_delta_clamp is not None:
            limit = abs(float(self.extension_delta_clamp))
            extension_clamped = torch.max(torch.abs(delta_extension), dim=-1).values > limit
            delta_extension = torch.clamp(delta_extension, -limit, limit)
        if has_delta_dp and self.dp_delta_clamp is not None:
            limit = abs(float(self.dp_delta_clamp))
            dp_clamped = torch.max(torch.abs(delta_dp), dim=-1).values > limit
            delta_dp = torch.clamp(delta_dp, -limit, limit)
        debug: dict[str, torch.Tensor | bool] = {
            "pred_norm": pred_norm.detach(),
            "pred_raw": pred_raw.detach(),
            "history_filled": self.history_filled.detach().clone(),
            "output_nan_or_inf": output_nan_or_inf.detach(),
            "extension_delta_clamped": extension_clamped.detach(),
            "dp_delta_clamped": dp_clamped.detach(),
            "use_force_input": self.use_force_input,
            "has_delta_dp_output": has_delta_dp,
            "target_mode": self.target_mode,
        }
        return torch.cat([delta_extension, delta_dp], dim=-1), debug

    def predict_from_feature_sequence(self, raw_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.set_history_from_sequence(raw_sequence)
        pred, debug = self._predict_from_history()
        return pred[:, 0:4], pred[:, 4:8], debug

    def step(
        self,
        *,
        valve_cmd: torch.Tensor,
        dP: torch.Tensor,
        extension: torch.Tensor,
        extension_velocity: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        force: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        features = self.assemble_features(
            valve_cmd=valve_cmd,
            dP=dP,
            extension=extension,
            extension_velocity=extension_velocity,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            force=force,
        )
        self.append_features(features)
        pred, debug = self._predict_from_history()
        debug["input_features"] = features.detach()
        debug["history_ready"] = self.history_filled >= self.history_length
        return pred[:, 0:4], pred[:, 4:8], debug
