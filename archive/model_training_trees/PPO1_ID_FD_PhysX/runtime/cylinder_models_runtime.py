from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

# Ensure sequence_cylinder_common can be imported (it uses absolute imports)
_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))


CheckpointKind = Literal["id", "rom", "fd"]


@dataclass(frozen=True)
class ROMStepOutput:
    delta_extension: torch.Tensor
    extension_velocity_next: torch.Tensor
    delta_dP: torch.Tensor
    raw_output: torch.Tensor


@dataclass(frozen=True)
class FDStepOutput:
    delta_extension: torch.Tensor
    delta_extension_velocity: torch.Tensor
    delta_dP: torch.Tensor
    raw_output: torch.Tensor


def resolve_latest_checkpoint(models_root: str | Path, kind: CheckpointKind, requested: str | Path = "latest") -> Path:
    requested_path = Path(str(requested)).expanduser()
    if str(requested) and str(requested) != "latest":
        if not requested_path.is_absolute():
            requested_path = Path(models_root) / requested_path
        resolved = requested_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    root = Path(models_root).expanduser().resolve()
    if kind == "id":
        candidates = list((root / "id_cylinder_space " / "runs_id_cylinder").glob("*/best_id_cylinder*.pt"))
    elif kind == "rom":
        candidates = list((root / "rom_cylinder" / "runs_rom_cylinder").glob("*/best_rom_cylinder.pt"))
    elif kind == "fd":
        candidates = list((root / "fd_cylinder_space " / "runs_fd_cylinder").glob("*/best_fd_cylinder.pt"))
    else:
        raise ValueError(f"Unsupported checkpoint kind: {kind!r}")

    non_smoke = [path for path in candidates if "smoke" not in {part.lower() for part in path.parts}]
    usable = non_smoke or candidates
    if not usable:
        raise FileNotFoundError(f"No {kind} checkpoint found under {root}")
    return max(usable, key=lambda path: path.stat().st_mtime).resolve()


def _ensure_import_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def _import_from_dir(path: Path, module_name: str):
    _ensure_import_path(path)
    return importlib.import_module(module_name)


class CylinderIDRuntime:
    """Runtime wrapper for latest cylinder-space ID checkpoints."""

    def __init__(
        self,
        *,
        models_root: str | Path,
        checkpoint: str | Path = "latest",
        num_envs: int,
        device: str | torch.device,
    ) -> None:
        self.models_root = Path(models_root).expanduser().resolve()
        self.checkpoint_path = resolve_latest_checkpoint(self.models_root, "id", checkpoint)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        self.num_envs = int(num_envs)

        id_dir = self.models_root / "id_cylinder_space "
        common = _import_from_dir(id_dir, "id_cylinder_common")
        ckpt = common.load_checkpoint(self.checkpoint_path, map_location="cpu")
        model_config = dict(ckpt["model_config"])
        self.model_type = str(model_config.get("model_type", ""))
        self.input_dim = int(model_config.get("input_dim", 24))
        self.target_mode = str(ckpt.get("target_mode", "normalized_valve_cmd"))
        self.input_feature_names = list(ckpt.get("input_feature_names", []))
        self.output_feature_names = list(ckpt.get("output_feature_names", []))
        self.command_limits = torch.as_tensor(
            ckpt.get("command_limits", [1.0, 1.0, 1.0, 1.0]),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 4)

        if self.model_type == "IDCylinderCNNLSTM":
            sequence_common = _import_from_dir(id_dir, "id_cylinder_sequence_common")
            self.model, self.checkpoint, self.normalizers = sequence_common.load_sequence_model_from_checkpoint(
                self.checkpoint_path,
                device=self.device,
            )
            self._sequence_state = self.model.initial_state(self.num_envs, device=self.device)
        elif self.model_type == "IDCylinderMLP":
            self.model, self.checkpoint, self.normalizers = common.load_model_from_checkpoint(
                self.checkpoint_path,
                device=self.device,
            )
            self._sequence_state = None
        else:
            raise ValueError(f"Unsupported cylinder ID model type: {self.model_type!r}")
        self.model.eval()
        self.last_input_shape: tuple[int, ...] | None = None
        self.last_output_shape: tuple[int, ...] | None = None
        self.last_input_raw: torch.Tensor | None = None
        self.last_input_normalized: torch.Tensor | None = None
        self.last_output_normalized: torch.Tensor | None = None
        self.last_output_denormalized: torch.Tensor | None = None
        self.last_delta_valve_cmd: torch.Tensor | None = None
        self.last_output_preclip: torch.Tensor | None = None
        self.last_output_clipped: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._sequence_state is None:
            return
        if env_ids is None:
            self._sequence_state = self.model.initial_state(self.num_envs, device=self.device)
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        for tensor in self._sequence_state:
            tensor[env_ids] = 0.0

    @torch.no_grad()
    def predict(self, id_input_raw: torch.Tensor, *, prev_valve_cmd: torch.Tensor) -> torch.Tensor:
        if id_input_raw.shape[-1] != self.input_dim:
            raise ValueError(
                f"ID input must have final dimension {self.input_dim}, got {tuple(id_input_raw.shape)}."
            )
        self.last_input_shape = tuple(id_input_raw.shape)
        x = id_input_raw.to(device=self.device, dtype=torch.float32)
        x_mean = self.normalizers["x_mean"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        x_std = self.normalizers["x_std"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        y_mean = self.normalizers["y_mean"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        y_std = self.normalizers["y_std"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        x_norm = (x - x_mean) / torch.clamp(x_std, min=1.0e-8)
        self.last_input_raw = x.detach().clone()
        self.last_input_normalized = x_norm.detach().clone()

        if self.model_type == "IDCylinderCNNLSTM":
            if self._sequence_state is None or self._sequence_state[0].shape[0] != x.shape[0]:
                self._sequence_state = self.model.initial_state(x.shape[0], device=self.device)
            y_norm, self._sequence_state = self.model.forward_step(x_norm, self._sequence_state)
        else:
            y_norm = self.model(x_norm)

        y_denorm = y_norm * y_std + y_mean
        valve_cmd = y_denorm
        self.last_output_normalized = y_norm.detach().clone()
        self.last_output_denormalized = y_denorm.detach().clone()
        self.last_delta_valve_cmd = y_denorm.detach().clone() if "delta" in self.target_mode else None
        if "delta" in self.target_mode:
            valve_cmd = prev_valve_cmd.to(device=self.device, dtype=torch.float32) + valve_cmd
        valve_cmd = torch.nan_to_num(valve_cmd, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_output_preclip = valve_cmd.detach().clone()
        valve_cmd = torch.clamp(valve_cmd, -self.command_limits, self.command_limits)
        self.last_output_clipped = valve_cmd.detach().clone()
        self.last_output_shape = tuple(valve_cmd.shape)
        return valve_cmd

    def input_z_score_summary(self, id_input_raw: torch.Tensor) -> dict[str, float]:
        x = id_input_raw.to(device=self.device, dtype=torch.float32)
        x_mean = self.normalizers["x_mean"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        x_std = self.normalizers["x_std"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        z = (x - x_mean) / torch.clamp(x_std, min=1.0e-8)
        abs_z = z.abs()
        return {
            "max_abs_z": float(abs_z.max().detach().cpu()),
            "mean_abs_z": float(abs_z.mean().detach().cpu()),
            "frac_abs_z_gt_3": float((abs_z > 3.0).float().mean().detach().cpu()),
            "frac_abs_z_gt_5": float((abs_z > 5.0).float().mean().detach().cpu()),
        }


class CylinderROMRuntime:
    """Runtime wrapper for the learned cylinder ROM velocity model."""

    def __init__(
        self,
        *,
        models_root: str | Path,
        checkpoint: str | Path = "latest",
        num_envs: int,
        device: str | torch.device,
    ) -> None:
        self.models_root = Path(models_root).expanduser().resolve()
        self.checkpoint_path = resolve_latest_checkpoint(self.models_root, "rom", checkpoint)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        self.num_envs = int(num_envs)

        rom_dir = self.models_root / "rom_cylinder"
        common = _import_from_dir(rom_dir, "rom_cylinder_common")
        self.model, self.checkpoint, self.normalizers = common.load_model_from_checkpoint(
            self.checkpoint_path,
            device=self.device,
        )
        self.model.eval()
        self.history_length = int(self.model.history_length)
        self.input_dim = int(self.model.input_dim)
        self.output_dim = int(self.model.output_dim)
        self.sample_time_s = float(self.checkpoint.get("sample_time_s", 0.05))
        self.history = torch.zeros(
            (self.num_envs, self.history_length, self.input_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_input_shape: tuple[int, ...] | None = None
        self.last_output_shape: tuple[int, ...] | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.history.zero_()
            return
        self.history[env_ids.to(device=self.device, dtype=torch.long)] = 0.0

    @torch.no_grad()
    def step(
        self,
        *,
        valve_cmd: torch.Tensor,
        extension: torch.Tensor,
        extension_velocity: torch.Tensor,
        dP: torch.Tensor,
        force: torch.Tensor,
        dt: float,
    ) -> ROMStepOutput:
        row = torch.cat([valve_cmd, extension, extension_velocity, dP], dim=-1)
        if self.input_dim == 20:
            row = torch.cat([row, force], dim=-1)
        if row.shape[-1] != self.input_dim:
            raise ValueError(f"ROM input row must have width {self.input_dim}, got {row.shape[-1]}.")
        row = torch.nan_to_num(row.to(device=self.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1, :] = row
        self.last_input_shape = tuple(self.history.shape)

        x_mean = self.normalizers["x_mean"].to(device=self.device, dtype=torch.float32)
        x_std = self.normalizers["x_std"].to(device=self.device, dtype=torch.float32)
        y_mean = self.normalizers["y_mean"].to(device=self.device, dtype=torch.float32)
        y_std = self.normalizers["y_std"].to(device=self.device, dtype=torch.float32)
        x_norm = (self.history - x_mean) / torch.clamp(x_std, min=1.0e-8)
        y_norm = self.model(x_norm)
        y_raw = y_norm * y_std + y_mean
        velocity_next = y_raw[:, 0:4]
        if y_raw.shape[-1] >= 8:
            delta_dP = y_raw[:, 4:8]
        else:
            delta_dP = torch.zeros_like(dP)
        delta_extension = velocity_next * float(dt)
        self.last_output_shape = tuple(y_raw.shape)
        return ROMStepOutput(
            delta_extension=torch.nan_to_num(delta_extension, nan=0.0, posinf=0.0, neginf=0.0),
            extension_velocity_next=torch.nan_to_num(velocity_next, nan=0.0, posinf=0.0, neginf=0.0),
            delta_dP=torch.nan_to_num(delta_dP, nan=0.0, posinf=0.0, neginf=0.0),
            raw_output=y_raw.detach(),
        )


class CylinderFDRuntime:
    """Optional one-step FD dynamics path: ID ValveCmd -> DeltaExtension."""

    def __init__(
        self,
        *,
        models_root: str | Path,
        checkpoint: str | Path = "latest",
        device: str | torch.device,
    ) -> None:
        self.models_root = Path(models_root).expanduser().resolve()
        self.checkpoint_path = resolve_latest_checkpoint(self.models_root, "fd", checkpoint)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")

        fd_dir = self.models_root / "fd_cylinder_space "
        common = _import_from_dir(fd_dir, "fd_cylinder_common")
        self.model, self.checkpoint, self.normalizers = common.load_model_from_checkpoint(
            self.checkpoint_path,
            device=self.device,
        )
        self.model.eval()
        self.sample_time_s = float(self.checkpoint.get("sample_time_s", 0.05))
        self.last_input_shape: tuple[int, ...] | None = None
        self.last_output_shape: tuple[int, ...] | None = None
        self.input_dim = int(self.checkpoint.get("model_config", {}).get("input_dim", 20))
        self.output_dim = int(self.checkpoint.get("model_config", {}).get("output_dim", 12))
        self.target_mode = str(self.checkpoint.get("target_mode", "normalized_delta_cylinder_state"))
        self.input_feature_names = list(self.checkpoint.get("input_feature_names", []))
        self.output_feature_names = list(self.checkpoint.get("output_feature_names", []))
        self.last_input_raw: torch.Tensor | None = None
        self.last_input_normalized: torch.Tensor | None = None
        self.last_output_normalized: torch.Tensor | None = None
        self.last_output_denormalized: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        return None

    @torch.no_grad()
    def step(
        self,
        *,
        valve_cmd: torch.Tensor,
        extension: torch.Tensor,
        extension_velocity: torch.Tensor,
        dP: torch.Tensor,
        force: torch.Tensor,
    ) -> FDStepOutput:
        row = torch.cat([valve_cmd, extension, extension_velocity, dP, force], dim=-1)
        if row.shape[-1] != self.input_dim:
            raise ValueError(f"FD input row must have width {self.input_dim}, got {row.shape[-1]}.")
        self.last_input_shape = tuple(row.shape)
        row = torch.nan_to_num(row.to(device=self.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x_mean = self.normalizers["x_mean"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        x_std = self.normalizers["x_std"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        y_mean = self.normalizers["y_mean"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        y_std = self.normalizers["y_std"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        x_norm = (row - x_mean) / torch.clamp(x_std, min=1.0e-8)
        y_norm = self.model(x_norm)
        y_raw = y_norm * y_std + y_mean
        self.last_input_raw = row.detach().clone()
        self.last_input_normalized = x_norm.detach().clone()
        self.last_output_normalized = y_norm.detach().clone()
        self.last_output_denormalized = y_raw.detach().clone()
        self.last_output_shape = tuple(y_raw.shape)
        return FDStepOutput(
            delta_extension=torch.nan_to_num(y_raw[:, 0:4], nan=0.0, posinf=0.0, neginf=0.0),
            delta_extension_velocity=torch.nan_to_num(y_raw[:, 4:8], nan=0.0, posinf=0.0, neginf=0.0),
            delta_dP=torch.nan_to_num(y_raw[:, 8:12], nan=0.0, posinf=0.0, neginf=0.0),
            raw_output=y_raw.detach(),
        )

    def input_z_score_summary(self, fd_input_raw: torch.Tensor) -> dict[str, float]:
        x = fd_input_raw.to(device=self.device, dtype=torch.float32)
        x_mean = self.normalizers["x_mean"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        x_std = self.normalizers["x_std"].to(device=self.device, dtype=torch.float32).reshape(1, -1)
        z = (x - x_mean) / torch.clamp(x_std, min=1.0e-8)
        abs_z = z.abs()
        return {
            "max_abs_z": float(abs_z.max().detach().cpu()),
            "mean_abs_z": float(abs_z.mean().detach().cpu()),
            "frac_abs_z_gt_3": float((abs_z > 3.0).float().mean().detach().cpu()),
            "frac_abs_z_gt_5": float((abs_z > 5.0).float().mean().detach().cpu()),
        }

@dataclass(frozen=True)
class SequenceIDStepOutput:
    delta_valve_cmd_raw: torch.Tensor  # [B, 4] denormalized delta (before clipping)
    delta_valve_cmd_normalized: torch.Tensor  # [B, 4]
    valve_cmd_preclip: torch.Tensor  # [B, 4] prev + delta before clipping
    valve_cmd: torch.Tensor  # [B, 4] final clipped [-limits, +limits]
    clipping_flags: torch.Tensor  # [B, 4] bool where clipping occurred


@dataclass(frozen=True)
class SequenceFDStepOutput:
    delta_extension_raw: torch.Tensor  # [B, 4] as output by FD model
    delta_extension_scaled: torch.Tensor  # [B, 4] after temporal scaling


class SequenceIDRuntime:
    """Runtime wrapper for the coupled HistoryGRU sequence ID checkpoint.

    Predicts DeltaValveCmd from a causal history of (Extension, ExtensionVelocity,
    PreviousValveCmd) plus current (TargetDeltaExtension, TargetDeltaExtensionVelocity,
    PreviousValveCmd).
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        num_envs: int,
        device: str | torch.device,
        verbose: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Sequence ID checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        self.num_envs = int(num_envs)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self._checkpoint = ckpt
        self.model_type = str(ckpt["model_config"]["model_type"])
        if self.model_type != "HistoryGRU":
            raise ValueError(
                f"SequenceIDRuntime expects HistoryGRU checkpoint, got {self.model_type!r}"
            )

        self.step_dim = int(ckpt["model_config"]["step_dim"])
        self.current_dim = int(ckpt["model_config"]["current_dim"])
        self.output_dim = int(ckpt["model_config"]["output_dim"])
        self.hidden_dim = int(ckpt["model_config"]["hidden_dim"])
        self.num_layers = int(ckpt["model_config"].get("num_layers", 1))
        self.history_length = int(ckpt.get("history_length", 10))
        self.sample_time_s = float(ckpt.get("sample_time_s", 0.05))

        self.step_feature_names = list(ckpt.get("step_feature_names", []))
        self.current_feature_names = list(ckpt.get("current_feature_names", []))
        self.target_feature_names = list(ckpt.get("target_feature_names", []))
        self.target_mode = str(ckpt.get("target_mode", "delta_valvecmd"))

        command_limits_raw = ckpt.get("command_limits", [1.0, 1.0, 1.0, 1.0])
        self.command_limits = torch.as_tensor(
            command_limits_raw, dtype=torch.float32, device=self.device
        ).reshape(1, 4)

        self.seq_mean = torch.as_tensor(
            ckpt["seq_mean"], dtype=torch.float32, device=self.device
        ).reshape(1, 1, -1)
        self.seq_std = torch.as_tensor(
            ckpt["seq_std"], dtype=torch.float32, device=self.device
        ).reshape(1, 1, -1)
        self.current_mean = torch.as_tensor(
            ckpt.get("current_mean", []), dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        self.current_std = torch.as_tensor(
            ckpt.get("current_std", []), dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        self.y_mean = torch.as_tensor(
            ckpt["y_mean"], dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        self.y_std = torch.as_tensor(
            ckpt["y_std"], dtype=torch.float32, device=self.device
        ).reshape(1, -1)

        # Build the HistoryGRU model
        from sequence_cylinder_common import HistoryGRU as _HistoryGRU_ID
        self.model = _HistoryGRU_ID(
            step_dim=self.step_dim,
            current_dim=self.current_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        if self.model.training:
            raise RuntimeError("Sequence ID model is still in training mode after eval().")

        # History buffer: [num_envs, H, step_dim]
        self._step_history = torch.zeros(
            (self.num_envs, self.history_length, self.step_dim),
            dtype=torch.float32,
            device=self.device,
        )

        # Diagnostic storage
        self.last_seq_raw: torch.Tensor | None = None
        self.last_seq_norm: torch.Tensor | None = None
        self.last_current_raw: torch.Tensor | None = None
        self.last_current_norm: torch.Tensor | None = None
        self.last_delta_valve_norm: torch.Tensor | None = None
        self.last_delta_valve_raw: torch.Tensor | None = None
        self.last_valve_cmd_preclip: torch.Tensor | None = None
        self.last_valve_cmd: torch.Tensor | None = None
        self.last_clipping_flags: torch.Tensor | None = None
        self.hidden_state_contract = "fixed_window_zero_initial_hidden"
        self.last_hidden_state_norm: float | None = None

        self._validate_feature_names()
        self._validate_tensor_contract()

        if verbose:
            print(
                "[SequenceIDRuntime] Loaded HistoryGRU: "
                f"H={self.history_length}, step_dim={self.step_dim}, cur_dim={self.current_dim}, "
                f"out_dim={self.output_dim}, hidden={self.hidden_dim}, layers={self.num_layers}",
                flush=True,
            )
            print(
                f"[SequenceIDRuntime] Step features: {self.step_feature_names}",
                flush=True,
            )
            print(
                f"[SequenceIDRuntime] Current features: {self.current_feature_names}",
                flush=True,
            )
            print(
                f"[SequenceIDRuntime] Target features: {self.target_feature_names}",
                flush=True,
            )
            print(
                f"[SequenceIDRuntime] Command limits: {command_limits_raw}",
                flush=True,
            )
            print(
                f"[SequenceIDRuntime] Sample time: {self.sample_time_s} s",
                flush=True,
            )

    def _validate_feature_names(self) -> None:
        expected_step = [
            "Extension_1", "Extension_2", "Extension_3", "Extension_4",
            "ExtensionVelocity_1", "ExtensionVelocity_2", "ExtensionVelocity_3", "ExtensionVelocity_4",
            "ValveCmd_1", "ValveCmd_2", "ValveCmd_3", "ValveCmd_4",
        ]
        expected_current = [
            "TargetDeltaExtension_1", "TargetDeltaExtension_2", "TargetDeltaExtension_3", "TargetDeltaExtension_4",
            "TargetDeltaExtensionVelocity_1", "TargetDeltaExtensionVelocity_2",
            "TargetDeltaExtensionVelocity_3", "TargetDeltaExtensionVelocity_4",
            "PreviousValveCmd_1", "PreviousValveCmd_2", "PreviousValveCmd_3", "PreviousValveCmd_4",
        ]
        if self.step_feature_names and self.step_feature_names != expected_step:
            raise RuntimeError(
                f"ID step_feature_names mismatch: expected {expected_step}, got {self.step_feature_names}"
            )
        if self.current_feature_names and self.current_feature_names != expected_current:
            raise RuntimeError(
                f"ID current_feature_names mismatch: expected {expected_current}, got {self.current_feature_names}"
            )
        expected_target = [
            "DeltaValveCmd_1", "DeltaValveCmd_2", "DeltaValveCmd_3", "DeltaValveCmd_4",
        ]
        if self.target_feature_names and self.target_feature_names != expected_target:
            raise RuntimeError(
                f"ID target_feature_names mismatch: expected {expected_target}, got {self.target_feature_names}"
            )
        if self.target_mode != "delta_valvecmd":
            raise RuntimeError(f"ID target_mode must be 'delta_valvecmd', got {self.target_mode!r}.")

    def _validate_tensor_contract(self) -> None:
        if self._step_history.shape != (self.num_envs, self.history_length, self.step_dim):
            raise RuntimeError(f"ID history shape mismatch: {tuple(self._step_history.shape)}")
        if self.seq_mean.shape[-1] != self.step_dim or self.seq_std.shape[-1] != self.step_dim:
            raise RuntimeError("ID sequence scaler width does not match step_dim.")
        if self.current_mean.shape[-1] != self.current_dim or self.current_std.shape[-1] != self.current_dim:
            raise RuntimeError("ID current scaler width does not match current_dim.")
        if self.y_mean.shape[-1] != self.output_dim or self.y_std.shape[-1] != self.output_dim:
            raise RuntimeError("ID output scaler width does not match output_dim.")
        if self.output_dim != 4:
            raise RuntimeError(f"ID output_dim must be 4, got {self.output_dim}.")
        if self.command_limits.shape != (1, 4):
            raise RuntimeError(f"ID command_limits shape mismatch: {tuple(self.command_limits.shape)}")
        for name, tensor in (
            ("seq_mean", self.seq_mean),
            ("seq_std", self.seq_std),
            ("current_mean", self.current_mean),
            ("current_std", self.current_std),
            ("y_mean", self.y_mean),
            ("y_std", self.y_std),
            ("command_limits", self.command_limits),
        ):
            if tensor.device != self.device:
                raise RuntimeError(f"ID {name} is on {tensor.device}, expected {self.device}.")
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"ID {name} contains NaN/Inf.")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._step_history.zero_()
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() and (int(env_ids.min().item()) < 0 or int(env_ids.max().item()) >= self.num_envs):
            raise IndexError(
                f"ID reset env_ids out of bounds for history batch {self.num_envs}: "
                f"min={int(env_ids.min().item())} max={int(env_ids.max().item())}"
            )
        self._step_history[env_ids] = 0.0

    def init_history_from_state(
        self,
        env_ids: torch.Tensor | None,
        extension: torch.Tensor,
        extension_velocity: torch.Tensor,
        previous_valve_cmd: torch.Tensor,
    ) -> None:
        """Fill all H history rows with the current state (used at reset)."""
        if env_ids is None:
            env_slice = slice(None)
        else:
            env_slice = env_ids.to(device=self.device, dtype=torch.long)
            if env_slice.numel() and (int(env_slice.min().item()) < 0 or int(env_slice.max().item()) >= self.num_envs):
                raise IndexError(
                    f"ID init_history env_ids out of bounds for history batch {self.num_envs}: "
                    f"min={int(env_slice.min().item())} max={int(env_slice.max().item())}"
                )
        for name, tensor in (
            ("extension", extension),
            ("extension_velocity", extension_velocity),
            ("previous_valve_cmd", previous_valve_cmd),
        ):
            if tensor.shape[0] != self.num_envs:
                raise ValueError(
                    f"ID init_history expects full-env {name} batch {self.num_envs}, got {tuple(tensor.shape)}."
                )
        row = torch.cat(
            [extension[env_slice], extension_velocity[env_slice], previous_valve_cmd[env_slice]],
            dim=-1,
        )
        if row.shape[-1] != self.step_dim:
            raise ValueError(
                f"History row dim {row.shape[-1]} != step_dim {self.step_dim}"
            )
        self._step_history[env_slice] = row.unsqueeze(1).expand(-1, self.history_length, -1)

    @torch.no_grad()
    def predict_delta_valve(
        self,
        target_delta_x: torch.Tensor,  # [B, 4]
        target_delta_xdot: torch.Tensor,  # [B, 4]
        previous_valve_cmd: torch.Tensor,  # [B, 4]
    ) -> SequenceIDStepOutput:
        """Predict DeltaValveCmd from history and current features.

        ID history contains Extension, ExtensionVelocity, PreviousValveCmd (causal:
        ValveCmd[k] not in history before ID inference).
        """
        B = target_delta_x.shape[0]
        if B != self._step_history.shape[0]:
            raise ValueError(f"ID batch size {B} does not match history env count {self._step_history.shape[0]}.")
        if self.model.training:
            raise RuntimeError("Sequence ID model must be in eval mode during inference.")
        if self._step_history.shape != (B, self.history_length, self.step_dim):
            raise RuntimeError(f"ID history shape mismatch before predict: {tuple(self._step_history.shape)}")
        if self._step_history.device != self.device:
            raise RuntimeError(f"ID history is on {self._step_history.device}, expected {self.device}.")
        history_before = self._step_history.detach().clone()

        # Build current features: [TargetDeltaExtension(4), TargetDeltaExtensionVelocity(4), PreviousValveCmd(4)]
        current_raw = torch.cat(
            [target_delta_x, target_delta_xdot, previous_valve_cmd], dim=-1
        ).to(device=self.device, dtype=torch.float32)
        current_raw = torch.nan_to_num(current_raw, nan=0.0, posinf=0.0, neginf=0.0)

        seq_raw = self._step_history.clone()

        # Normalize
        seq_norm = (seq_raw - self.seq_mean) / torch.clamp(self.seq_std, min=1.0e-8)
        cur_norm = (current_raw - self.current_mean) / torch.clamp(self.current_std, min=1.0e-8)
        if not torch.isfinite(seq_norm).all() or not torch.isfinite(cur_norm).all():
            raise FloatingPointError("Sequence ID normalized inputs contain NaN/Inf.")

        # Store diagnostics
        self.last_seq_raw = seq_raw.detach().clone()
        self.last_seq_norm = seq_norm.detach().clone()
        self.last_current_raw = current_raw.detach().clone()
        self.last_current_norm = cur_norm.detach().clone()

        # Forward
        y_norm = self.model(seq_norm, cur_norm)
        if y_norm.shape != (B, 4):
            raise RuntimeError(f"Sequence ID output shape mismatch: {tuple(y_norm.shape)}")
        if y_norm.device != self.device:
            raise RuntimeError(f"Sequence ID output is on {y_norm.device}, expected {self.device}.")
        if not torch.isfinite(y_norm).all():
            raise FloatingPointError("Sequence ID normalized output contains NaN/Inf.")
        y_raw = y_norm * self.y_std + self.y_mean  # DeltaValveCmd denormalized
        if not torch.isfinite(y_raw).all():
            raise FloatingPointError("Sequence ID denormalized output contains NaN/Inf.")
        if not torch.equal(self._step_history, history_before):
            raise RuntimeError("Sequence ID predict_delta_valve mutated history; this would leak causality.")
        self.last_hidden_state_norm = 0.0

        self.last_delta_valve_norm = y_norm.detach().clone()
        self.last_delta_valve_raw = y_raw.detach().clone()

        # Compute valve_cmd = clip(previous_valve_cmd + delta_valve_cmd, limits)
        valve_cmd_preclip = previous_valve_cmd.to(device=self.device, dtype=torch.float32) + y_raw
        valve_cmd_preclip = torch.nan_to_num(valve_cmd_preclip, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_valve_cmd_preclip = valve_cmd_preclip.detach().clone()

        valve_cmd = torch.clamp(valve_cmd_preclip, -self.command_limits, self.command_limits)
        clipping_flags = (valve_cmd != valve_cmd_preclip)

        self.last_valve_cmd = valve_cmd.detach().clone()
        self.last_clipping_flags = clipping_flags.detach().clone()

        return SequenceIDStepOutput(
            delta_valve_cmd_raw=y_raw,
            delta_valve_cmd_normalized=y_norm,
            valve_cmd_preclip=valve_cmd_preclip,
            valve_cmd=valve_cmd,
            clipping_flags=clipping_flags,
        )

    def push_history(
        self,
        extension: torch.Tensor,  # [B, 4]
        extension_velocity: torch.Tensor,  # [B, 4]
        valve_cmd: torch.Tensor,  # [B, 4] — previous_valve_cmd for future ID steps
    ) -> None:
        """After PhysX step: push new state into rolling history window."""
        row = torch.cat([extension, extension_velocity, valve_cmd], dim=-1)
        row = torch.nan_to_num(row.to(device=self.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if row.shape[-1] != self.step_dim:
            raise ValueError(f"Push row dim {row.shape[-1]} != step_dim {self.step_dim}")
        self._step_history = torch.roll(self._step_history, shifts=-1, dims=1)
        self._step_history[:, -1, :] = row
        if self._step_history.shape != (self.num_envs, self.history_length, self.step_dim):
            raise RuntimeError(f"ID history shape mismatch after push: {tuple(self._step_history.shape)}")


class SequenceFDRuntime:
    """Runtime wrapper for the coupled HistoryGRU sequence FD checkpoint.

    Predicts DeltaExtension from a history of (ValveCmd, Extension, ExtensionVelocity)
    with the current ValveCmd in the final history row.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        num_envs: int,
        device: str | torch.device,
        verbose: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Sequence FD checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        self.num_envs = int(num_envs)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self._checkpoint = ckpt
        self.model_type = str(ckpt["model_config"]["model_type"])
        if self.model_type != "HistoryGRU":
            raise ValueError(
                f"SequenceFDRuntime expects HistoryGRU checkpoint, got {self.model_type!r}"
            )

        self.step_dim = int(ckpt["model_config"]["step_dim"])
        self.current_dim = int(ckpt["model_config"]["current_dim"])
        self.output_dim = int(ckpt["model_config"]["output_dim"])
        self.hidden_dim = int(ckpt["model_config"]["hidden_dim"])
        self.num_layers = int(ckpt["model_config"].get("num_layers", 1))
        self.history_length = int(ckpt.get("history_length", 10))
        self.sample_time_s = float(ckpt.get("sample_time_s", 0.05))

        self.input_feature_names = list(ckpt.get("input_feature_names", []))
        self.target_feature_names = list(ckpt.get("target_feature_names", []))
        self.target_mode = str(ckpt.get("target_mode", "delta_extension"))

        self.seq_mean = torch.as_tensor(
            ckpt["seq_mean"], dtype=torch.float32, device=self.device
        ).reshape(1, 1, -1)
        self.seq_std = torch.as_tensor(
            ckpt["seq_std"], dtype=torch.float32, device=self.device
        ).reshape(1, 1, -1)
        self.y_mean = torch.as_tensor(
            ckpt["y_mean"], dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        self.y_std = torch.as_tensor(
            ckpt["y_std"], dtype=torch.float32, device=self.device
        ).reshape(1, -1)

        # Build the HistoryGRU model
        from sequence_cylinder_common import HistoryGRU as _HistoryGRU_ID
        self.model = _HistoryGRU_ID(
            step_dim=self.step_dim,
            current_dim=self.current_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        if self.model.training:
            raise RuntimeError("Sequence FD model is still in training mode after eval().")

        # History buffer: [num_envs, H, step_dim]
        # Step dim = [ValveCmd(4), Extension(4), ExtensionVelocity(4)]
        self._step_history = torch.zeros(
            (self.num_envs, self.history_length, self.step_dim),
            dtype=torch.float32,
            device=self.device,
        )

        # Diagnostic storage
        self.last_seq_raw: torch.Tensor | None = None
        self.last_seq_norm: torch.Tensor | None = None
        self.last_delta_extension_norm: torch.Tensor | None = None
        self.last_delta_extension_raw: torch.Tensor | None = None
        self.last_delta_extension_scaled: torch.Tensor | None = None
        self.hidden_state_contract = "fixed_window_zero_initial_hidden"
        self.last_hidden_state_norm: float | None = None

        self._validate_feature_names()
        self._validate_tensor_contract()

        if verbose:
            print(
                "[SequenceFDRuntime] Loaded HistoryGRU: "
                f"H={self.history_length}, step_dim={self.step_dim}, cur_dim={self.current_dim}, "
                f"out_dim={self.output_dim}, hidden={self.hidden_dim}, layers={self.num_layers}",
                flush=True,
            )
            print(
                f"[SequenceFDRuntime] Input features: {self.input_feature_names}",
                flush=True,
            )
            print(
                f"[SequenceFDRuntime] Target features: {self.target_feature_names}",
                flush=True,
            )
            print(
                f"[SequenceFDRuntime] Sample time: {self.sample_time_s} s",
                flush=True,
            )

    def _validate_feature_names(self) -> None:
        expected_input = [
            "ValveCmd_1", "ValveCmd_2", "ValveCmd_3", "ValveCmd_4",
            "Extension_1", "Extension_2", "Extension_3", "Extension_4",
            "ExtensionVelocity_1", "ExtensionVelocity_2", "ExtensionVelocity_3", "ExtensionVelocity_4",
        ]
        if self.input_feature_names and self.input_feature_names != expected_input:
            raise RuntimeError(
                f"FD input_feature_names mismatch: expected {expected_input}, got {self.input_feature_names}"
            )
        expected_target = [
            "DeltaExtension_next_1", "DeltaExtension_next_2",
            "DeltaExtension_next_3", "DeltaExtension_next_4",
        ]
        if self.target_feature_names and self.target_feature_names != expected_target:
            raise RuntimeError(
                f"FD target_feature_names mismatch: expected {expected_target}, got {self.target_feature_names}"
            )
        if self.target_mode != "delta_extension":
            raise RuntimeError(f"FD target_mode must be 'delta_extension', got {self.target_mode!r}.")

    def _validate_tensor_contract(self) -> None:
        if self._step_history.shape != (self.num_envs, self.history_length, self.step_dim):
            raise RuntimeError(f"FD history shape mismatch: {tuple(self._step_history.shape)}")
        if self.seq_mean.shape[-1] != self.step_dim or self.seq_std.shape[-1] != self.step_dim:
            raise RuntimeError("FD sequence scaler width does not match step_dim.")
        if self.y_mean.shape[-1] != self.output_dim or self.y_std.shape[-1] != self.output_dim:
            raise RuntimeError("FD output scaler width does not match output_dim.")
        if self.current_dim != 0:
            raise RuntimeError(f"FD current_dim must be 0, got {self.current_dim}.")
        if self.output_dim != 4:
            raise RuntimeError(f"FD output_dim must be 4, got {self.output_dim}.")
        for name, tensor in (
            ("seq_mean", self.seq_mean),
            ("seq_std", self.seq_std),
            ("y_mean", self.y_mean),
            ("y_std", self.y_std),
        ):
            if tensor.device != self.device:
                raise RuntimeError(f"FD {name} is on {tensor.device}, expected {self.device}.")
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"FD {name} contains NaN/Inf.")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._step_history.zero_()
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() and (int(env_ids.min().item()) < 0 or int(env_ids.max().item()) >= self.num_envs):
            raise IndexError(
                f"FD reset env_ids out of bounds for history batch {self.num_envs}: "
                f"min={int(env_ids.min().item())} max={int(env_ids.max().item())}"
            )
        self._step_history[env_ids] = 0.0

    def init_history_from_state(
        self,
        env_ids: torch.Tensor | None,
        valve_cmd: torch.Tensor,
        extension: torch.Tensor,
        extension_velocity: torch.Tensor,
    ) -> None:
        """Fill all H history rows with the current state (used at reset)."""
        if env_ids is None:
            env_slice = slice(None)
        else:
            env_slice = env_ids.to(device=self.device, dtype=torch.long)
            if env_slice.numel() and (int(env_slice.min().item()) < 0 or int(env_slice.max().item()) >= self.num_envs):
                raise IndexError(
                    f"FD init_history env_ids out of bounds for history batch {self.num_envs}: "
                    f"min={int(env_slice.min().item())} max={int(env_slice.max().item())}"
                )
        for name, tensor in (
            ("valve_cmd", valve_cmd),
            ("extension", extension),
            ("extension_velocity", extension_velocity),
        ):
            if tensor.shape[0] != self.num_envs:
                raise ValueError(
                    f"FD init_history expects full-env {name} batch {self.num_envs}, got {tuple(tensor.shape)}."
                )
        row = torch.cat(
            [valve_cmd[env_slice], extension[env_slice], extension_velocity[env_slice]],
            dim=-1,
        )
        if row.shape[-1] != self.step_dim:
            raise ValueError(
                f"FD history row dim {row.shape[-1]} != step_dim {self.step_dim}"
            )
        self._step_history[env_slice] = row.unsqueeze(1).expand(-1, self.history_length, -1)

    def push_current_state(
        self,
        valve_cmd: torch.Tensor,  # [B, 4] current ValveCmd[k]
        extension: torch.Tensor,  # [B, 4] current Extension[k]
        extension_velocity: torch.Tensor,  # [B, 4] current ExtensionVelocity[k]
    ) -> None:
        """Push current ValveCmd + state into the final FD history row.

        Called AFTER ID inference produces valve_cmd[k] and BEFORE FD prediction,
        so FD sees the current command in its final history row.
        """
        B = valve_cmd.shape[0]
        if B != self._step_history.shape[0]:
            raise ValueError(f"FD batch size {B} does not match history env count {self._step_history.shape[0]}.")
        row = torch.cat([valve_cmd, extension, extension_velocity], dim=-1)
        row = torch.nan_to_num(row.to(device=self.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if row.shape[-1] != self.step_dim:
            raise ValueError(f"FD push row dim {row.shape[-1]} != step_dim {self.step_dim}")
        self._step_history = torch.roll(self._step_history, shifts=-1, dims=1)
        self._step_history[:, -1, :] = row
        if self._step_history.shape != (self.num_envs, self.history_length, self.step_dim):
            raise RuntimeError(f"FD history shape mismatch after push: {tuple(self._step_history.shape)}")

    @torch.no_grad()
    def predict_delta_extension(self) -> SequenceFDStepOutput:
        """Predict DeltaExtension from the current FD history buffer."""
        return self.predict_delta_extension_from_history(self._step_history)

    @torch.no_grad()
    def predict_delta_extension_from_history(self, step_history: torch.Tensor) -> SequenceFDStepOutput:
        """Predict DeltaExtension from a supplied FD history without mutating runtime history."""
        if self.model.training:
            raise RuntimeError("Sequence FD model must be in eval mode during inference.")
        if step_history.shape != (self.num_envs, self.history_length, self.step_dim):
            raise RuntimeError(f"FD history shape mismatch before predict: {tuple(step_history.shape)}")
        if step_history.device != self.device:
            raise RuntimeError(f"FD history is on {step_history.device}, expected {self.device}.")
        seq_raw = step_history.clone()
        seq_norm = (seq_raw - self.seq_mean) / torch.clamp(self.seq_std, min=1.0e-8)
        if not torch.isfinite(seq_norm).all():
            raise FloatingPointError("Sequence FD normalized input contains NaN/Inf.")

        self.last_seq_raw = seq_raw.detach().clone()
        self.last_seq_norm = seq_norm.detach().clone()

        # FD has no current features (current_dim=0), so pass None
        y_norm = self.model(seq_norm, None)
        if y_norm.shape != (self.num_envs, 4):
            raise RuntimeError(f"Sequence FD output shape mismatch: {tuple(y_norm.shape)}")
        if y_norm.device != self.device:
            raise RuntimeError(f"Sequence FD output is on {y_norm.device}, expected {self.device}.")
        if not torch.isfinite(y_norm).all():
            raise FloatingPointError("Sequence FD normalized output contains NaN/Inf.")
        y_raw = y_norm * self.y_std + self.y_mean
        y_raw = torch.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)

        self.last_delta_extension_norm = y_norm.detach().clone()
        self.last_delta_extension_raw = y_raw.detach().clone()
        self.last_delta_extension_scaled = y_raw.detach().clone()
        self.last_hidden_state_norm = 0.0

        return SequenceFDStepOutput(
            delta_extension_raw=y_raw,
            delta_extension_scaled=y_raw,  # caller scales by control_dt/model_dt
        )
