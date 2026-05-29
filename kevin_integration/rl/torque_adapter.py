from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from kevin_integration.rl.action_adapter import clamp_and_sanitize


TORQUE_ADAPTER_SCALE_PRESETS = {
    "aggressive": [0.001, 0.03, 0.003, 0.0005],
    "moderate": [0.0003, 0.003, 0.001, 0.0002],
    "conservative": [0.0001, 0.001, 0.0005, 0.0001],
}
DEFAULT_TORQUE_ADAPTER_PRESET = "conservative"
DEFAULT_TORQUE_ADAPTER_SCALE = TORQUE_ADAPTER_SCALE_PRESETS[DEFAULT_TORQUE_ADAPTER_PRESET]
DEFAULT_TORQUE_ADAPTER_BIAS = [0.0, 0.0, 0.0, 0.0]
TORQUE_ADAPTER_MODES = ("none", "scale_bias", "tanh_squash", "residual_pd")
TORQUE_ADAPTER_PRESETS = tuple(TORQUE_ADAPTER_SCALE_PRESETS.keys())


@dataclass(frozen=True)
class TorqueAdapterCfg:
    mode: str = "scale_bias"
    preset: str = DEFAULT_TORQUE_ADAPTER_PRESET
    scale: Sequence[float] = tuple(DEFAULT_TORQUE_ADAPTER_SCALE)
    bias: Sequence[float] = tuple(DEFAULT_TORQUE_ADAPTER_BIAS)
    fd_torque_scale: float = 1.0e6
    use_fd_residual_alpha: bool = True
    fd_residual_alpha: float = 0.002
    max_abs_torque: float = 5.0
    pd_kp: Sequence[float] | None = None
    pd_kd: Sequence[float] | None = None


def parse_torque_adapter_scale(value: str | Sequence[float] | None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("torque_adapter_scale must contain exactly 4 comma-separated values")
        scale = [float(part) for part in parts]
    else:
        scale = [float(part) for part in value]

    if len(scale) != 4:
        raise ValueError(f"torque_adapter_scale must contain exactly 4 values, got {len(scale)}")
    return scale


def resolve_torque_adapter_scale(preset: str, override: str | Sequence[float] | None = None) -> list[float]:
    if preset not in TORQUE_ADAPTER_SCALE_PRESETS:
        raise ValueError(f"Unknown torque adapter preset {preset!r}. Expected one of {TORQUE_ADAPTER_PRESETS}.")
    parsed = parse_torque_adapter_scale(override)
    if parsed is not None:
        return parsed
    return list(TORQUE_ADAPTER_SCALE_PRESETS[preset])


def effective_torque_adapter_scale(scale: Sequence[float], fd_residual_alpha: float, use_fd_residual_alpha: bool) -> list[float]:
    gain = float(fd_residual_alpha) if use_fd_residual_alpha else 1.0
    return [gain * float(value) for value in scale]


def apply_fd_residual_authority(
    tau_fd_adapted: torch.Tensor,
    *,
    fd_residual_alpha: float,
    use_fd_residual_alpha: bool = True,
) -> torch.Tensor:
    tau_fd_adapted = clamp_and_sanitize(tau_fd_adapted, None)
    if not use_fd_residual_alpha:
        return tau_fd_adapted
    return clamp_and_sanitize(float(fd_residual_alpha) * tau_fd_adapted, None)


def _as_joint_tensor(values: Sequence[float] | torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=reference.dtype, device=reference.device).reshape(1, -1)
    if tensor.shape[-1] != reference.shape[-1]:
        raise ValueError(f"{name} must have {reference.shape[-1]} values, got {tensor.shape[-1]}")
    return tensor


def _scale_bias(
    fd_simscape_output: torch.Tensor,
    *,
    scale: Sequence[float] | torch.Tensor,
    bias: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    scale_t = _as_joint_tensor(scale, fd_simscape_output, "torque_adapter_scale")
    bias_t = _as_joint_tensor(bias, fd_simscape_output, "torque_adapter_bias")
    return scale_t * fd_simscape_output + bias_t


def _pd_torque(
    fd_simscape_output: torch.Tensor,
    *,
    q: torch.Tensor | None,
    qdot: torch.Tensor | None,
    q_ref: torch.Tensor | None,
    pd_kp: Sequence[float] | torch.Tensor | None,
    pd_kd: Sequence[float] | torch.Tensor | None,
) -> torch.Tensor:
    if q is None or qdot is None:
        return torch.zeros_like(fd_simscape_output)

    q_cmd = q[..., : fd_simscape_output.shape[-1]]
    qdot_cmd = qdot[..., : fd_simscape_output.shape[-1]]
    if q_ref is None:
        q_ref_cmd = q_cmd
    else:
        q_ref_cmd = q_ref[..., : fd_simscape_output.shape[-1]].to(device=fd_simscape_output.device)

    kp = torch.zeros_like(fd_simscape_output) if pd_kp is None else _as_joint_tensor(pd_kp, fd_simscape_output, "pd_kp")
    kd = torch.zeros_like(fd_simscape_output) if pd_kd is None else _as_joint_tensor(pd_kd, fd_simscape_output, "pd_kd")
    return kp * (q_ref_cmd - q_cmd) - kd * qdot_cmd


def adapt_fd_to_isaac_torque(
    fd_simscape_output: torch.Tensor,
    *,
    mode: str = "scale_bias",
    scale: Sequence[float] | torch.Tensor = DEFAULT_TORQUE_ADAPTER_SCALE,
    bias: Sequence[float] | torch.Tensor = DEFAULT_TORQUE_ADAPTER_BIAS,
    fd_torque_scale: float = 1.0e6,
    fd_residual_alpha: float = 0.002,
    max_abs_torque: float = 5.0,
    q: torch.Tensor | None = None,
    qdot: torch.Tensor | None = None,
    q_ref: torch.Tensor | None = None,
    pd_kp: Sequence[float] | torch.Tensor | None = None,
    pd_kd: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Map FD Simscape-domain output into Isaac joint effort units.

    The learned FD output is not applied directly to Isaac. This adapter is the
    explicit place where Simscape-domain actuator/generalized outputs are scaled
    into the much smaller Isaac effort range.
    """
    fd_simscape_output = clamp_and_sanitize(fd_simscape_output, None)
    if fd_simscape_output.shape[-1] != 4:
        raise ValueError(f"fd_simscape_output must have last dimension 4, got {fd_simscape_output.shape[-1]}")

    if mode == "none":
        tau = fd_simscape_output
    elif mode == "scale_bias":
        tau = _scale_bias(fd_simscape_output, scale=scale, bias=bias)
    elif mode == "tanh_squash":
        fd_scale = max(float(fd_torque_scale), 1.0e-8)
        tau = float(max_abs_torque) * torch.tanh(fd_simscape_output / fd_scale)
    elif mode == "residual_pd":
        tau_pd = _pd_torque(
            fd_simscape_output,
            q=q,
            qdot=qdot,
            q_ref=q_ref,
            pd_kp=pd_kp,
            pd_kd=pd_kd,
        )
        residual = _scale_bias(fd_simscape_output, scale=scale, bias=bias)
        tau = tau_pd + float(fd_residual_alpha) * residual
    else:
        raise ValueError(f"Unknown torque adapter mode {mode!r}. Expected one of {TORQUE_ADAPTER_MODES}.")

    return clamp_and_sanitize(tau, None)


def lowpass_torque(target: torch.Tensor, previous: torch.Tensor, alpha: float) -> torch.Tensor:
    alpha = min(1.0, max(0.0, float(alpha)))
    return previous + alpha * (target - previous)


def rate_limit_torque(target: torch.Tensor, previous: torch.Tensor, rate_limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    limit = max(float(rate_limit), 0.0)
    if limit <= 0.0:
        limited = previous
        fraction = torch.mean((torch.abs(target - previous) > 0.0).float(), dim=-1)
        return limited, fraction

    delta = target - previous
    limited_delta = torch.clamp(delta, -limit, limit)
    limited = previous + limited_delta
    fraction = torch.mean((torch.abs(delta) > limit).float(), dim=-1)
    return limited, fraction
