from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from kevin_integration.controllers.learned_models import (
    AAMModel,
    ForwardDynamicsModel,
    InverseDynamicsModel,
    SystemDynamicsModel,
)
from kevin_integration.rl.action_adapter import clamp_and_sanitize
from kevin_integration.rl.observation_builder import build_aam_input, build_fd_input, build_id_input, build_sd_input
from kevin_integration.rl.torque_adapter import (
    DEFAULT_TORQUE_ADAPTER_PRESET,
    DEFAULT_TORQUE_ADAPTER_BIAS,
    DEFAULT_TORQUE_ADAPTER_SCALE,
    adapt_fd_to_isaac_torque,
    apply_fd_residual_authority,
    lowpass_torque,
    rate_limit_torque,
)


class TorchSequenceBuffer:
    def __init__(self, *, num_envs: int, window_size: int, feature_dim: int, device: torch.device):
        self.data = torch.zeros((num_envs, window_size, feature_dim), dtype=torch.float32, device=device)

    def append(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] != self.data.shape[0] or value.shape[-1] != self.data.shape[-1]:
            raise ValueError(
                f"Expected value shape ({self.data.shape[0]}, {self.data.shape[-1]}), got {tuple(value.shape)}"
            )
        value = value.detach()
        self.data = torch.roll(self.data, shifts=-1, dims=1)
        self.data[:, -1, :] = value
        return self.data

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.data.zero_()
            return
        self.data[env_ids] = 0.0

    def fill(self, value: torch.Tensor, env_ids: Optional[torch.Tensor] = None) -> None:
        if value.shape[-1] != self.data.shape[-1]:
            raise ValueError(f"Expected fill value dim {self.data.shape[-1]}, got {value.shape[-1]}")
        value = value.detach().to(device=self.data.device, dtype=self.data.dtype)
        if env_ids is None:
            if value.shape[0] != self.data.shape[0]:
                raise ValueError(f"Expected {self.data.shape[0]} fill rows, got {value.shape[0]}")
            self.data[:] = value.reshape(value.shape[0], 1, value.shape[-1])
            return
        env_ids = env_ids.to(device=self.data.device, dtype=torch.long)
        if value.shape[0] != env_ids.numel():
            raise ValueError(f"Expected {env_ids.numel()} fill rows, got {value.shape[0]}")
        self.data[env_ids] = value.reshape(value.shape[0], 1, value.shape[-1])


@dataclass(frozen=True)
class AdaptiveRLLearnedControllerCfg:
    models_dir: str | Path
    num_envs: int
    device: str
    fallback_window: int = 10
    activation: str = "relu"
    dropout: float = 0.0
    max_abs_valve: float = 1.0
    max_abs_torque: float = 5.0
    torque_adapter_mode: str = "scale_bias"
    torque_adapter_preset: str = DEFAULT_TORQUE_ADAPTER_PRESET
    torque_adapter_scale: tuple[float, float, float, float] = tuple(DEFAULT_TORQUE_ADAPTER_SCALE)
    torque_adapter_bias: tuple[float, float, float, float] = tuple(DEFAULT_TORQUE_ADAPTER_BIAS)
    fd_torque_scale: float = 1.0e6
    use_fd_residual_alpha: bool = True
    fd_residual_alpha: float = 0.002
    torque_rate_limit: float = 1.0
    torque_lowpass_alpha: float = 0.2
    use_torque_rate_limit: bool = True
    use_torque_lowpass: bool = True
    max_abs_pressure_delta: float = 1.0e7
    max_abs_fnet: float = 1.0e7
    sd_output_mode: str = "state"
    sd_nan_guard: bool = True
    sd_clamp_outputs: bool = True


@dataclass(frozen=True)
class SDObserverOutput:
    state_pred: torch.Tensor
    q_next_pred: torch.Tensor
    qdot_next_pred: torch.Tensor
    dP_next_pred: torch.Tensor
    fnet_next_pred: torch.Tensor
    output_nan_or_inf: torch.Tensor
    output_clamped_fraction: torch.Tensor
    debug: dict[str, torch.Tensor]


@dataclass(frozen=True)
class AdaptiveRLControlOutput:
    torque: torch.Tensor
    fd_simscape_output: torch.Tensor
    tau_fd_adapted: torch.Tensor
    tau_applied_raw: torch.Tensor
    tau_isaac_filtered: torch.Tensor
    tau_isaac_clamped: torch.Tensor
    torque_clamp_fraction: torch.Tensor
    torque_adapter_clamp_fraction: torch.Tensor
    torque_rate_limited_fraction: torch.Tensor
    torque_adapter_mode: str
    torque_adapter_preset: str
    use_fd_residual_alpha: bool
    fd_residual_alpha: float
    valve_cmd: torch.Tensor
    z_arm_hat: torch.Tensor
    qddot_cmd: torch.Tensor
    dP_next: torch.Tensor
    fnet_next: torch.Tensor
    sd_state: torch.Tensor
    sd_debug: dict[str, torch.Tensor]


class AdaptiveRLLearnedController:
    def __init__(self, cfg: AdaptiveRLLearnedControllerCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        models_dir = Path(cfg.models_dir)

        self.aam = AAMModel(
            checkpoint_path=models_dir / "aam.pth",
            activation=cfg.activation,
            dropout=cfg.dropout,
            map_location="cpu",
            device=str(self.device),
        )
        self.id = InverseDynamicsModel(
            checkpoint_path=models_dir / "id.pth",
            activation=cfg.activation,
            dropout=cfg.dropout,
            map_location="cpu",
            device=str(self.device),
        )
        self.fd = ForwardDynamicsModel(
            checkpoint_path=models_dir / "fd.pth",
            scaler_path=models_dir / "fd_scaler.pth",
            activation=cfg.activation,
            dropout=cfg.dropout,
            map_location="cpu",
            device=str(self.device),
        )
        self.sd = SystemDynamicsModel(
            checkpoint_path=models_dir / "sd.pth",
            activation=cfg.activation,
            dropout=cfg.dropout,
            map_location="cpu",
            device=str(self.device),
        )
        self._freeze_model(self.aam)
        self._freeze_model(self.id)
        self._freeze_model(self.fd)
        self._freeze_model(self.sd)

        aam_window = self.aam.window_size or cfg.fallback_window
        id_window = self.id.window_size or cfg.fallback_window
        fd_window = self.fd.window_size or cfg.fallback_window
        sd_window = self.sd.window_size or cfg.fallback_window

        self._aam_history = TorchSequenceBuffer(
            num_envs=cfg.num_envs, window_size=aam_window, feature_dim=18, device=self.device
        )
        self._id_history = TorchSequenceBuffer(
            num_envs=cfg.num_envs, window_size=id_window, feature_dim=19, device=self.device
        )
        self._fd_history = TorchSequenceBuffer(
            num_envs=cfg.num_envs, window_size=fd_window, feature_dim=18, device=self.device
        )
        self._sd_history = TorchSequenceBuffer(
            num_envs=cfg.num_envs, window_size=sd_window, feature_dim=22, device=self.device
        )
        self._prev_tau_isaac = torch.zeros((cfg.num_envs, 4), dtype=torch.float32, device=self.device)
        self._last_id_input_raw = torch.zeros((cfg.num_envs, 19), dtype=torch.float32, device=self.device)
        self._last_id_input_normalized = torch.zeros_like(self._last_id_input_raw)
        self._last_z_arm_hat = torch.zeros((cfg.num_envs, 24), dtype=torch.float32, device=self.device)
        self._id_qddot_std = self._extract_id_qddot_std()
        if self.aam.window_size is None and self.id.window_size is not None and aam_window != self.id.window_size:
            self._aam_history = TorchSequenceBuffer(
                num_envs=cfg.num_envs, window_size=self.id.window_size, feature_dim=18, device=self.device
            )
            aam_window = self.id.window_size
        self.runtime_contract = {
            "aam_input": "[q5, qdot5, dP4, prev_valve4]",
            "aam_window": aam_window,
            "aam_output": "z_arm_hat[24]",
            "id_input": "[q5, qdot5, qddot_cmd5, dP4]",
            "id_window": id_window,
            "id_output": "ValveCmd[4]",
            "id_qddot_std": self._id_qddot_std.detach().cpu().tolist(),
            "fd_input": "[q5, qdot5, dP4, ValveCmd4]",
            "fd_window": fd_window,
            "fd_output": "TauHat[4] Simscape-domain torque/load estimate",
            "fd_output_units": "checkpoint/scaler denormalized Simscape torque units",
            "fd_effort_actuation": "disabled in ROM position-target mode",
            "sd_window": sd_window,
        }

    @staticmethod
    def _freeze_model(wrapper) -> None:
        wrapper.model.requires_grad_(False)
        wrapper.model.eval()

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        self._aam_history.reset(env_ids)
        self._id_history.reset(env_ids)
        self._fd_history.reset(env_ids)
        self._sd_history.reset(env_ids)
        if env_ids is None:
            self._prev_tau_isaac.zero_()
        else:
            self._prev_tau_isaac[env_ids] = 0.0

    def warm_start_histories(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        prev_valve_cmd: torch.Tensor,
        *,
        qddot_cmd: torch.Tensor | None = None,
        fnet: torch.Tensor | None = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = q.to(self.device)
        qdot = qdot.to(self.device)
        dP = dP.to(self.device)
        prev_valve_cmd = prev_valve_cmd.to(self.device)
        if qddot_cmd is None:
            qddot_cmd = torch.zeros((q.shape[0], 5), dtype=torch.float32, device=self.device)
        else:
            qddot_cmd = qddot_cmd.to(self.device)
        if fnet is None:
            fnet = torch.zeros_like(dP)
        else:
            fnet = fnet.to(self.device)

        aam_x_t = build_aam_input(q, qdot, dP, prev_valve_cmd)
        id_x_t = build_id_input(q, qdot, qddot_cmd, dP)
        fd_x_t = build_fd_input(q, qdot, dP, prev_valve_cmd)
        sd_x_t = build_sd_input(q, qdot, dP, fnet, prev_valve_cmd)
        self._aam_history.fill(aam_x_t, env_ids)
        self._id_history.fill(id_x_t, env_ids)
        self._fd_history.fill(fd_x_t, env_ids)
        self._sd_history.fill(sd_x_t, env_ids)
        return clamp_and_sanitize(self.aam(self._aam_history.data if env_ids is None else self._aam_history.data[env_ids]), None)

    def compute_z_arm(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        prev_valve_cmd: torch.Tensor,
        *,
        update_history: bool = True,
    ) -> torch.Tensor:
        aam_x_t = build_aam_input(q, qdot, dP, prev_valve_cmd).to(self.device)
        aam_seq = self._aam_history.append(aam_x_t) if update_history else self._aam_history.data
        return clamp_and_sanitize(self.aam(aam_seq), None)

    def run_fd_observer(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        valve_cmd: torch.Tensor,
    ) -> torch.Tensor:
        q = q.to(self.device)
        qdot = qdot.to(self.device)
        dP = dP.to(self.device)
        valve_cmd = valve_cmd.to(self.device)
        fd_x_t = build_fd_input(q, qdot, dP, valve_cmd)
        fd_seq = self._fd_history.append(fd_x_t)
        return clamp_and_sanitize(self.fd(fd_seq), None)

    def fd_output_normalizer(self, *, like: torch.Tensor) -> torch.Tensor:
        std = getattr(self.fd, "y_std", None)
        if std is None:
            return torch.ones((1, 4), dtype=like.dtype, device=like.device)
        std_tensor = torch.as_tensor(std, dtype=like.dtype, device=like.device).reshape(1, -1)
        if std_tensor.shape[-1] != 4:
            raise RuntimeError(f"FD y_std must have four outputs, got {std_tensor.shape[-1]}.")
        return torch.clamp(std_tensor, min=1.0e-8)

    def _record_id_debug(self, id_x_t: torch.Tensor, z_arm_hat: torch.Tensor) -> None:
        raw = id_x_t.detach().to(device=self.device, dtype=torch.float32)
        self._last_id_input_raw[: raw.shape[0]] = raw
        if getattr(self.id, "feature_mean", None) is not None and getattr(self.id, "feature_std", None) is not None:
            normalized = self.id.normalize_features(raw)
        else:
            normalized = raw
        self._last_id_input_normalized[: raw.shape[0]] = normalized.detach().to(device=self.device, dtype=torch.float32)
        self._last_z_arm_hat[: z_arm_hat.shape[0]] = z_arm_hat.detach().to(device=self.device, dtype=torch.float32)

    def debug_state(self) -> dict[str, torch.Tensor]:
        return {
            "id_input_raw": self._last_id_input_raw.detach().clone(),
            "id_input_normalized": self._last_id_input_normalized.detach().clone(),
            "id_input_normalized_min": torch.min(self._last_id_input_normalized.detach(), dim=-1).values,
            "id_input_normalized_max": torch.max(self._last_id_input_normalized.detach(), dim=-1).values,
            "id_input_normalized_abs_max": torch.max(torch.abs(self._last_id_input_normalized.detach()), dim=-1).values,
            "z_arm_norm": torch.linalg.norm(self._last_z_arm_hat.detach(), dim=-1),
            "z_arm_std": torch.std(self._last_z_arm_hat.detach(), dim=-1, unbiased=False),
        }

    def _extract_id_qddot_std(self) -> torch.Tensor:
        std = getattr(self.id, "feature_std", None)
        if std is None:
            return torch.ones((4,), dtype=torch.float32, device=self.device)
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=self.device).reshape(-1)
        if std_tensor.numel() < 15:
            raise RuntimeError("ID feature_std must contain qddot feature entries 10:15.")
        qddot_std = torch.clamp(std_tensor[10:14], min=1.0e-6)
        if not torch.isfinite(qddot_std).all():
            raise RuntimeError("ID qddot feature std contains NaN/Inf.")
        return qddot_std

    def _resolve_qddot_vector(
        self,
        value: float | str | Sequence[float],
        *,
        name: str,
        like: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(value, str):
            if value != "id_qddot_std":
                raise ValueError(f"{name} string value must be 'id_qddot_std', got {value!r}.")
            return self._id_qddot_std.to(device=like.device, dtype=like.dtype).reshape(1, 4)
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device).reshape(-1)
            if tensor.numel() == 5:
                tensor = tensor[:4]
            if tensor.numel() != 4:
                raise ValueError(f"{name} sequence must contain four values, got {tensor.numel()}.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN/Inf.")
            return torch.clamp(tensor, min=1.0e-8).reshape(1, 4)
        scalar = float(value)
        if not torch.isfinite(torch.tensor(scalar)):
            raise ValueError(f"{name} must be finite.")
        return torch.full((1, 4), max(abs(scalar), 1.0e-8), dtype=like.dtype, device=like.device)

    def action_scale_vector(self, scale: float | str | Sequence[float], *, like: torch.Tensor) -> torch.Tensor:
        return self._resolve_qddot_vector(scale, name="action_to_qddot_scale", like=like)

    def qddot_to_action(self, qddot_cmd_4d: torch.Tensor, scale: float | str | Sequence[float]) -> torch.Tensor:
        qddot_cmd_4d = qddot_cmd_4d.to(self.device)
        if qddot_cmd_4d.shape[-1] != 4:
            raise ValueError(f"qddot_cmd_4d must have last dimension 4, got {qddot_cmd_4d.shape[-1]}")
        scale_vec = self.action_scale_vector(scale, like=qddot_cmd_4d)
        return clamp_and_sanitize(qddot_cmd_4d / scale_vec, None)

    def build_qddot_cmd(
        self,
        action: torch.Tensor,
        scale: float | str | Sequence[float],
        clip: float | str | Sequence[float],
    ) -> torch.Tensor:
        action = action.to(self.device)
        if action.shape[-1] != 4:
            raise ValueError(f"action must have last dimension 4, got {action.shape[-1]}")
        scale_vec = self._resolve_qddot_vector(scale, name="action_to_qddot_scale", like=action)
        clip_vec = self._resolve_qddot_vector(clip, name="qddot_cmd_clip", like=action)
        qddot_cmd_4d = torch.clamp(clamp_and_sanitize(scale_vec * action, None), -clip_vec, clip_vec)
        zero_tool_joint = torch.zeros_like(qddot_cmd_4d[..., :1])
        return torch.cat([qddot_cmd_4d, zero_tool_joint], dim=-1)

    def apply_action(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        fnet: torch.Tensor,
        rl_action: torch.Tensor,
        *,
        z_arm_hat: torch.Tensor,
    ) -> AdaptiveRLControlOutput:
        return self.apply_action_full_pipeline(
            q,
            qdot,
            dP,
            fnet,
            rl_action,
            z_arm_hat=z_arm_hat,
            use_sd_feedback=True,
        )

    def apply_action_valve_only(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        rl_action: torch.Tensor,
        prev_valve_cmd: torch.Tensor | None = None,
        *,
        z_arm_hat: torch.Tensor | None = None,
        action_to_qddot_scale: float = 1.0,
        qddot_cmd_clip: float = 5.0,
    ) -> AdaptiveRLControlOutput:
        q = q.to(self.device)
        qdot = qdot.to(self.device)
        dP = dP.to(self.device)
        rl_action = rl_action.to(self.device)
        qddot_cmd = self.build_qddot_cmd(rl_action, action_to_qddot_scale, qddot_cmd_clip)
        if z_arm_hat is None:
            if prev_valve_cmd is None:
                raise ValueError("prev_valve_cmd is required when z_arm_hat is not provided.")
            z_arm_hat = self.compute_z_arm(q, qdot, dP, prev_valve_cmd.to(self.device))
        else:
            z_arm_hat = clamp_and_sanitize(z_arm_hat.to(self.device), None)

        id_x_t = build_id_input(q, qdot, qddot_cmd, dP)
        self._record_id_debug(id_x_t, z_arm_hat)
        id_seq = self._id_history.append(id_x_t)
        valve_cmd = clamp_and_sanitize(self.id(id_seq, z_arm_hat), self.cfg.max_abs_valve)

        torque = torch.zeros_like(valve_cmd)
        torque_clamp_fraction = torch.zeros((q.shape[0],), dtype=torch.float32, device=self.device)
        dP_next = torch.zeros_like(dP)
        fnet_next = torch.zeros_like(valve_cmd)
        sd_state = torch.zeros((q.shape[0], 18), dtype=torch.float32, device=self.device)
        sd_debug = self._zero_sd_debug(q.shape[0])
        return AdaptiveRLControlOutput(
            torque=torque,
            fd_simscape_output=torque,
            tau_fd_adapted=torque,
            tau_applied_raw=torque,
            tau_isaac_filtered=torque,
            tau_isaac_clamped=torque,
            torque_clamp_fraction=torque_clamp_fraction,
            torque_adapter_clamp_fraction=torque_clamp_fraction,
            torque_rate_limited_fraction=torque_clamp_fraction,
            torque_adapter_mode=self.cfg.torque_adapter_mode,
            torque_adapter_preset=self.cfg.torque_adapter_preset,
            use_fd_residual_alpha=self.cfg.use_fd_residual_alpha,
            fd_residual_alpha=self.cfg.fd_residual_alpha,
            valve_cmd=valve_cmd,
            z_arm_hat=z_arm_hat,
            qddot_cmd=qddot_cmd,
            dP_next=dP_next,
            fnet_next=fnet_next,
            sd_state=sd_state,
            sd_debug=sd_debug,
        )

    def apply_action_full_pipeline(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        fnet: torch.Tensor,
        rl_action: torch.Tensor,
        prev_valve_cmd: torch.Tensor | None = None,
        *,
        z_arm_hat: torch.Tensor | None = None,
        action_to_qddot_scale: float = 1.0,
        qddot_cmd_clip: float = 5.0,
        use_sd_feedback: bool = False,
    ) -> AdaptiveRLControlOutput:
        q = q.to(self.device)
        qdot = qdot.to(self.device)
        dP = dP.to(self.device)
        fnet = fnet.to(self.device)
        rl_action = rl_action.to(self.device)
        qddot_cmd = self.build_qddot_cmd(rl_action, action_to_qddot_scale, qddot_cmd_clip)
        if z_arm_hat is None:
            if prev_valve_cmd is None:
                raise ValueError("prev_valve_cmd is required when z_arm_hat is not provided.")
            z_arm_hat = self.compute_z_arm(q, qdot, dP, prev_valve_cmd.to(self.device))
        else:
            z_arm_hat = clamp_and_sanitize(z_arm_hat.to(self.device), None)

        id_x_t = build_id_input(q, qdot, qddot_cmd, dP)
        self._record_id_debug(id_x_t, z_arm_hat)
        id_seq = self._id_history.append(id_x_t)
        valve_cmd = clamp_and_sanitize(self.id(id_seq, z_arm_hat), self.cfg.max_abs_valve)

        fd_x_t = build_fd_input(q, qdot, dP, valve_cmd)
        fd_seq = self._fd_history.append(fd_x_t)
        fd_simscape_output = clamp_and_sanitize(self.fd(fd_seq), None)
        tau_fd_adapted = adapt_fd_to_isaac_torque(
            fd_simscape_output,
            mode=self.cfg.torque_adapter_mode,
            scale=self.cfg.torque_adapter_scale,
            bias=self.cfg.torque_adapter_bias,
            fd_torque_scale=self.cfg.fd_torque_scale,
            fd_residual_alpha=self.cfg.fd_residual_alpha,
            max_abs_torque=self.cfg.max_abs_torque,
            q=q,
            qdot=qdot,
        )
        tau_applied_raw = apply_fd_residual_authority(
            tau_fd_adapted,
            fd_residual_alpha=self.cfg.fd_residual_alpha,
            use_fd_residual_alpha=self.cfg.use_fd_residual_alpha,
        )
        torque_limit = max(float(self.cfg.max_abs_torque), 1.0e-8)
        torque_adapter_clamp_fraction = torch.mean((torch.abs(tau_fd_adapted) > torque_limit).float(), dim=-1)

        tau_isaac_filtered = tau_applied_raw
        torque_rate_limited_fraction = torch.zeros((q.shape[0],), dtype=torch.float32, device=self.device)
        if self.cfg.use_torque_lowpass:
            tau_isaac_filtered = lowpass_torque(
                tau_isaac_filtered,
                self._prev_tau_isaac[: q.shape[0]],
                self.cfg.torque_lowpass_alpha,
            )
            tau_isaac_filtered = clamp_and_sanitize(tau_isaac_filtered, None)
        if self.cfg.use_torque_rate_limit:
            tau_isaac_filtered, torque_rate_limited_fraction = rate_limit_torque(
                tau_isaac_filtered,
                self._prev_tau_isaac[: q.shape[0]],
                self.cfg.torque_rate_limit,
            )
            tau_isaac_filtered = clamp_and_sanitize(tau_isaac_filtered, None)

        tau_isaac_clamped = clamp_and_sanitize(tau_isaac_filtered, self.cfg.max_abs_torque)
        torque_clamp_fraction = torch.mean((torch.abs(tau_isaac_filtered) > torque_limit).float(), dim=-1)
        self._prev_tau_isaac[: q.shape[0]] = tau_isaac_clamped.detach()

        if use_sd_feedback:
            sd_observer = self.run_sd_observer(q, qdot, dP, fnet, valve_cmd, z_arm_hat)
            sd_state = sd_observer.state_pred
            dP_next = sd_observer.dP_next_pred
            fnet_next = sd_observer.fnet_next_pred
            sd_debug = sd_observer.debug
        else:
            sd_state = torch.zeros((q.shape[0], 18), dtype=torch.float32, device=self.device)
            dP_next = torch.zeros_like(dP)
            fnet_next = torch.zeros_like(fnet)
            sd_debug = self._zero_sd_debug(q.shape[0])

        return AdaptiveRLControlOutput(
            torque=tau_isaac_clamped,
            fd_simscape_output=fd_simscape_output,
            tau_fd_adapted=tau_fd_adapted,
            tau_applied_raw=tau_applied_raw,
            tau_isaac_filtered=tau_isaac_filtered,
            tau_isaac_clamped=tau_isaac_clamped,
            torque_clamp_fraction=torque_clamp_fraction,
            torque_adapter_clamp_fraction=torque_adapter_clamp_fraction,
            torque_rate_limited_fraction=torque_rate_limited_fraction,
            torque_adapter_mode=self.cfg.torque_adapter_mode,
            torque_adapter_preset=self.cfg.torque_adapter_preset,
            use_fd_residual_alpha=self.cfg.use_fd_residual_alpha,
            fd_residual_alpha=self.cfg.fd_residual_alpha,
            valve_cmd=valve_cmd,
            z_arm_hat=z_arm_hat,
            qddot_cmd=qddot_cmd,
            dP_next=dP_next,
            fnet_next=fnet_next,
            sd_state=sd_state,
            sd_debug=sd_debug,
        )

    def run_sd_observer(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        fnet: torch.Tensor,
        valve_cmd: torch.Tensor,
        z_arm_hat: torch.Tensor,
    ) -> SDObserverOutput:
        q = q.to(self.device)
        qdot = qdot.to(self.device)
        dP = dP.to(self.device)
        fnet = fnet.to(self.device)
        valve_cmd = valve_cmd.to(self.device)
        z_arm_hat = clamp_and_sanitize(z_arm_hat.to(self.device), None)

        sd_x_t = build_sd_input(q, qdot, dP, fnet, valve_cmd)
        sd_seq = self._sd_history.append(sd_x_t)
        sd_raw = self.sd(sd_seq, z_arm_hat)
        if sd_raw.shape[-1] != 18:
            raise RuntimeError(f"SD output must have last dimension 18, got {sd_raw.shape[-1]}")

        raw_nan_or_inf = (~torch.isfinite(sd_raw).all(dim=-1)).float()
        if self.cfg.sd_nan_guard:
            sd_raw = torch.nan_to_num(sd_raw, nan=0.0, posinf=0.0, neginf=0.0)

        state_pred = self._decode_sd_state(q, qdot, dP, fnet, sd_raw)
        decoded_nan_or_inf = (~torch.isfinite(state_pred).all(dim=-1)).float()
        output_nan_or_inf = torch.maximum(raw_nan_or_inf, decoded_nan_or_inf)
        if self.cfg.sd_nan_guard:
            state_pred = torch.nan_to_num(state_pred, nan=0.0, posinf=0.0, neginf=0.0)
            invalid = output_nan_or_inf.bool().reshape(-1, 1)
            state_pred = torch.where(invalid, torch.zeros_like(state_pred), state_pred)

        state_pred, clamped_fraction = self._clamp_sd_state(state_pred)
        q_next_pred = state_pred[..., 0:5]
        qdot_next_pred = state_pred[..., 5:10]
        dP_next_pred = state_pred[..., 10:14]
        fnet_next_pred = state_pred[..., 14:18]
        debug = {
            "sd_q_next_pred": q_next_pred.detach(),
            "sd_qdot_next_pred": qdot_next_pred.detach(),
            "sd_dP_next_pred": dP_next_pred.detach(),
            "sd_fnet_next_pred": fnet_next_pred.detach(),
            "sd_dP_mean": torch.mean(torch.abs(dP_next_pred.detach()), dim=-1),
            "sd_dP_abs_max": torch.max(torch.abs(dP_next_pred.detach()), dim=-1).values,
            "sd_fnet_mean": torch.mean(torch.abs(fnet_next_pred.detach()), dim=-1),
            "sd_fnet_abs_max": torch.max(torch.abs(fnet_next_pred.detach()), dim=-1).values,
            "sd_output_nan_or_inf": output_nan_or_inf.detach(),
            "sd_output_clamped_fraction": clamped_fraction.detach(),
        }
        return SDObserverOutput(
            state_pred=state_pred,
            q_next_pred=q_next_pred,
            qdot_next_pred=qdot_next_pred,
            dP_next_pred=dP_next_pred,
            fnet_next_pred=fnet_next_pred,
            output_nan_or_inf=output_nan_or_inf,
            output_clamped_fraction=clamped_fraction,
            debug=debug,
        )

    def _zero_sd_debug(self, batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "sd_q_next_pred": torch.zeros((batch_size, 5), dtype=torch.float32, device=self.device),
            "sd_qdot_next_pred": torch.zeros((batch_size, 5), dtype=torch.float32, device=self.device),
            "sd_dP_next_pred": torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device),
            "sd_fnet_next_pred": torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device),
            "sd_dP_mean": torch.zeros((batch_size,), dtype=torch.float32, device=self.device),
            "sd_dP_abs_max": torch.zeros((batch_size,), dtype=torch.float32, device=self.device),
            "sd_fnet_mean": torch.zeros((batch_size,), dtype=torch.float32, device=self.device),
            "sd_fnet_abs_max": torch.zeros((batch_size,), dtype=torch.float32, device=self.device),
            "sd_output_nan_or_inf": torch.zeros((batch_size,), dtype=torch.float32, device=self.device),
            "sd_output_clamped_fraction": torch.zeros((batch_size,), dtype=torch.float32, device=self.device),
        }

    def _decode_sd_state(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        fnet: torch.Tensor,
        sd_output: torch.Tensor,
    ) -> torch.Tensor:
        current_state = torch.cat([q, qdot, dP, fnet], dim=-1)
        target_mode = getattr(self.sd, "target_mode", "")
        if target_mode in ("delta_normalized", "normalized_delta"):
            return current_state + self.sd.denormalize_delta(sd_output)
        if target_mode in ("delta", "state_delta"):
            return current_state + sd_output
        if target_mode in ("state", "absolute_state", "next_state"):
            return sd_output
        if target_mode not in ("", "none"):
            raise ValueError(f"Unsupported SD target_mode in checkpoint metadata: {target_mode!r}")

        if self.cfg.sd_output_mode == "delta":
            return current_state + sd_output
        if self.cfg.sd_output_mode == "state":
            return sd_output
        raise ValueError("sd_output_mode must be 'delta' or 'state'")

    def _clamp_sd_state(self, state_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state_pred.shape[-1] != 18:
            raise RuntimeError(f"SD state prediction must have last dimension 18, got {state_pred.shape[-1]}")
        if not self.cfg.sd_clamp_outputs:
            fraction = torch.zeros((state_pred.shape[0],), dtype=torch.float32, device=self.device)
            return state_pred, fraction

        dP_pred = state_pred[..., 10:14]
        fnet_pred = state_pred[..., 14:18]
        dP_limit = max(float(self.cfg.max_abs_pressure_delta), 1.0e-8)
        fnet_limit = max(float(self.cfg.max_abs_fnet), 1.0e-8)
        dP_clamped_mask = torch.abs(dP_pred) > dP_limit
        fnet_clamped_mask = torch.abs(fnet_pred) > fnet_limit
        clamped_fraction = torch.mean(torch.cat([dP_clamped_mask, fnet_clamped_mask], dim=-1).float(), dim=-1)

        state_clamped = state_pred.clone()
        state_clamped[..., 10:14] = clamp_and_sanitize(dP_pred, self.cfg.max_abs_pressure_delta)
        state_clamped[..., 14:18] = clamp_and_sanitize(fnet_pred, self.cfg.max_abs_fnet)
        return state_clamped, clamped_fraction

    def _extract_hydraulic_state(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        fnet: torch.Tensor,
        sd_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sd_state.shape[-1] != 18:
            raise RuntimeError(f"SD output must have last dimension 18, got {sd_state.shape[-1]}")

        sd_state = self._decode_sd_state(q, qdot, dP, fnet, sd_state)
        sd_state, _ = self._clamp_sd_state(sd_state)

        dP_next = sd_state[..., 10:14]
        fnet_next = sd_state[..., 14:18]
        return dP_next, fnet_next
