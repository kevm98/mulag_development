from __future__ import annotations

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

    def build_qddot_cmd(self, action: torch.Tensor, scale: float, clip: float) -> torch.Tensor:
        action = action.to(self.device)
        if action.shape[-1] != 4:
            raise ValueError(f"action must have last dimension 4, got {action.shape[-1]}")
        qddot_cmd_4d = clamp_and_sanitize(scale * action, clip)
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
        id_seq = self._id_history.append(id_x_t)
        valve_cmd = clamp_and_sanitize(self.id(id_seq, z_arm_hat), self.cfg.max_abs_valve)

        torque = torch.zeros_like(valve_cmd)
        torque_clamp_fraction = torch.zeros((q.shape[0],), dtype=torch.float32, device=self.device)
        dP_next = torch.zeros_like(dP)
        fnet_next = torch.zeros_like(valve_cmd)
        sd_state = torch.zeros((q.shape[0], 18), dtype=torch.float32, device=self.device)
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
            sd_x_t = build_sd_input(q, qdot, dP, fnet, valve_cmd)
            sd_seq = self._sd_history.append(sd_x_t)
            sd_state = clamp_and_sanitize(self.sd(sd_seq, z_arm_hat), None)
            dP_next, fnet_next = self._extract_hydraulic_state(q, qdot, dP, fnet, sd_state)
        else:
            sd_state = torch.zeros((q.shape[0], 18), dtype=torch.float32, device=self.device)
            dP_next = torch.zeros_like(dP)
            fnet_next = torch.zeros_like(fnet)

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
        )

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

        if self.cfg.sd_output_mode == "delta":
            current_state = torch.cat([q, qdot, dP, fnet], dim=-1)
            sd_state = current_state + sd_state
        elif self.cfg.sd_output_mode != "state":
            raise ValueError("sd_output_mode must be 'delta' or 'state'")

        dP_next = clamp_and_sanitize(sd_state[..., 10:14], self.cfg.max_abs_pressure_delta)
        fnet_next = clamp_and_sanitize(sd_state[..., 14:18], self.cfg.max_abs_fnet)
        return dP_next, fnet_next
