from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch


@dataclass(frozen=True)
class ReducedHydraulicActuatorROMCfg:
    """Safe placeholder parameters for a reduced hydraulic cylinder ROM.

    These values are intentionally conservative bootstrap parameters. They are
    not final identified parameters and should be fitted from Simscape/full-order
    data before using this mode for training decisions.
    """

    output_mode: str = "velocity_to_position_target"
    Abar: tuple[float, float, float, float] = (1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3)
    Vt: tuple[float, float, float, float] = (1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3)
    E: tuple[float, float, float, float] = (1.0e8, 1.0e8, 1.0e8, 1.0e8)
    Kq: tuple[float, float, float, float] = (1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8)
    CL: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    mu: tuple[float, float, float, float] = (10.0, 10.0, 10.0, 10.0)
    Fs: tuple[float, float, float, float] = (10.0, 10.0, 10.0, 10.0)
    kf: tuple[float, float, float, float] = (100.0, 100.0, 100.0, 100.0)
    Pmax: tuple[float, float, float, float] = (1.0e7, 1.0e7, 1.0e7, 1.0e7)
    xdot_max: tuple[float, float, float, float] = (0.02, 0.02, 0.02, 0.02)
    dx_max: tuple[float, float, float, float] = (5.0e-4, 5.0e-4, 5.0e-4, 5.0e-4)
    delay_steps: tuple[int, int, int, int] = (0, 0, 0, 0)
    udz_pos: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    udz_neg: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    velocity_gain: tuple[float, float, float, float] = (0.02, 0.02, 0.02, 0.02)
    force_scale: tuple[float, float, float, float] = (1000.0, 1000.0, 1000.0, 1000.0)
    nan_guard: bool = True
    clamp_outputs: bool = True


@dataclass(frozen=True)
class ReducedHydraulicActuatorROMOutput:
    xdot_cyl_pred: torch.Tensor
    dx_cyl_pred: torch.Tensor
    P_L_next: torch.Tensor
    dP_next: torch.Tensor
    valve_cmd_delayed: torch.Tensor
    valve_cmd_effective: torch.Tensor
    debug: dict[str, torch.Tensor | str]


class ReducedHydraulicActuatorROM:
    """Reduced hydraulic cylinder ROM that outputs stroke increments.

    The ROM follows an explicit safe structure:
    delay -> asymmetric dead zone -> simple flow map -> pressure update ->
    saturated cylinder velocity/stroke increment. It does not command or return
    cylinder force for Isaac. The intended integration is to add dx to a
    prismatic cylinder position target and call set_joint_position_target.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        cfg: ReducedHydraulicActuatorROMCfg | None = None,
        num_cylinders: int = 4,
        device: str | torch.device = "cpu",
    ):
        self.cfg = cfg or ReducedHydraulicActuatorROMCfg()
        self.num_envs = int(num_envs)
        self.num_cylinders = int(num_cylinders)
        self.device = torch.device(device)
        if self.cfg.output_mode != "velocity_to_position_target":
            raise ValueError(
                "ReducedHydraulicActuatorROM only supports output_mode='velocity_to_position_target' "
                f"for this safe integration, got {self.cfg.output_mode!r}."
            )

        self.Abar = self._param(self.cfg.Abar, "Abar")
        self.Vt = torch.clamp(self._param(self.cfg.Vt, "Vt"), min=1.0e-9)
        self.E = self._param(self.cfg.E, "E")
        self.Kq = self._param(self.cfg.Kq, "Kq")
        self.CL = self._param(self.cfg.CL, "CL")
        self.mu = self._param(self.cfg.mu, "mu")
        self.Fs = self._param(self.cfg.Fs, "Fs")
        self.kf = self._param(self.cfg.kf, "kf")
        self.Pmax = torch.clamp(self._param(self.cfg.Pmax, "Pmax"), min=1.0e-8)
        self.xdot_max = torch.clamp(self._param(self.cfg.xdot_max, "xdot_max"), min=0.0)
        self.dx_max = torch.clamp(self._param(self.cfg.dx_max, "dx_max"), min=0.0)
        self.udz_pos = torch.clamp(self._param(self.cfg.udz_pos, "udz_pos"), min=0.0)
        self.udz_neg = torch.clamp(self._param(self.cfg.udz_neg, "udz_neg"), min=0.0)
        self.velocity_gain = torch.clamp(self._param(self.cfg.velocity_gain, "velocity_gain"), min=0.0)
        self.force_scale = torch.clamp(self._param(self.cfg.force_scale, "force_scale"), min=1.0e-8)

        delay = torch.as_tensor(self.cfg.delay_steps, dtype=torch.long, device=self.device)
        if delay.numel() != self.num_cylinders:
            raise ValueError(f"delay_steps must contain {self.num_cylinders} values, got {delay.numel()}")
        self.delay_steps = torch.clamp(delay, min=0)
        self.max_delay = int(torch.max(self.delay_steps).item()) if self.delay_steps.numel() else 0
        self._delay_buffer = torch.zeros(
            (self.num_envs, self.max_delay + 1, self.num_cylinders), dtype=torch.float32, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._delay_buffer.zero_()
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._delay_buffer[env_ids] = 0.0

    def step(
        self,
        *,
        valve_cmd: torch.Tensor,
        x_cyl: torch.Tensor,
        xdot_cyl: torch.Tensor,
        P_L: torch.Tensor,
        dt: float,
        Fnet: torch.Tensor | None = None,
        z_arm: torch.Tensor | None = None,
    ) -> ReducedHydraulicActuatorROMOutput:
        del x_cyl, Fnet, z_arm  # Reserved for future identified modular blocks.
        valve_cmd = self._shape(valve_cmd, "valve_cmd")
        xdot_cyl = self._shape(xdot_cyl, "xdot_cyl")
        P_L = self._shape(P_L, "P_L")
        dt = max(float(dt), 0.0)

        raw_invalid = (
            ~torch.isfinite(valve_cmd).all(dim=-1)
            | ~torch.isfinite(xdot_cyl).all(dim=-1)
            | ~torch.isfinite(P_L).all(dim=-1)
        )
        if self.cfg.nan_guard:
            valve_cmd = torch.nan_to_num(valve_cmd, nan=0.0, posinf=0.0, neginf=0.0)
            xdot_cyl = torch.nan_to_num(xdot_cyl, nan=0.0, posinf=0.0, neginf=0.0)
            P_L = torch.nan_to_num(P_L, nan=0.0, posinf=0.0, neginf=0.0)

        valve_delayed = self._delayed_valve(valve_cmd)
        valve_effective = self._deadzone(valve_delayed)

        Q_L = self.Kq * valve_effective
        pressure_gain = 4.0 * self.E / self.Vt
        P_L_unclamped = P_L + dt * pressure_gain * (Q_L - self.Abar * xdot_cyl - self.CL * P_L)
        pressure_clamp_mask = torch.abs(P_L_unclamped) > self.Pmax
        P_L_next = torch.clamp(P_L_unclamped, -self.Pmax, self.Pmax) if self.cfg.clamp_outputs else P_L_unclamped

        force_like = self.Abar * P_L_next - self.mu * xdot_cyl - self.Fs * torch.tanh(self.kf * xdot_cyl)
        xdot_unclamped = self.velocity_gain * torch.tanh(force_like / self.force_scale)
        xdot_clamp_mask = torch.abs(xdot_unclamped) > self.xdot_max
        xdot_pred = torch.clamp(xdot_unclamped, -self.xdot_max, self.xdot_max) if self.cfg.clamp_outputs else xdot_unclamped
        dx_unclamped = dt * xdot_pred
        dx_clamp_mask = torch.abs(dx_unclamped) > self.dx_max
        dx_pred = torch.clamp(dx_unclamped, -self.dx_max, self.dx_max) if self.cfg.clamp_outputs else dx_unclamped

        output_invalid = (
            ~torch.isfinite(P_L_next).all(dim=-1)
            | ~torch.isfinite(xdot_pred).all(dim=-1)
            | ~torch.isfinite(dx_pred).all(dim=-1)
        )
        nan_guard_triggered = raw_invalid | output_invalid
        if self.cfg.nan_guard:
            invalid = nan_guard_triggered.reshape(-1, 1)
            P_L_next = torch.where(invalid, torch.zeros_like(P_L_next), torch.nan_to_num(P_L_next))
            xdot_pred = torch.where(invalid, torch.zeros_like(xdot_pred), torch.nan_to_num(xdot_pred))
            dx_pred = torch.where(invalid, torch.zeros_like(dx_pred), torch.nan_to_num(dx_pred))

        output_zeroed_fraction = torch.mean((torch.abs(dx_pred) <= 1.0e-12).float(), dim=-1)
        debug: dict[str, torch.Tensor | str] = {
            "rom_enabled": "True",
            "rom_output_mode": self.cfg.output_mode,
            "rom_valve_cmd_mean": torch.mean(torch.abs(valve_cmd.detach()), dim=-1),
            "rom_valve_cmd_abs_max": torch.max(torch.abs(valve_cmd.detach()), dim=-1).values,
            "rom_dP_mean": torch.mean(torch.abs(P_L_next.detach()), dim=-1),
            "rom_P_L_mean": torch.mean(P_L_next.detach(), dim=-1),
            "rom_P_L_abs_max": torch.max(torch.abs(P_L_next.detach()), dim=-1).values,
            "rom_xdot_pred_mean": torch.mean(torch.abs(xdot_pred.detach()), dim=-1),
            "rom_xdot_pred_abs_max": torch.max(torch.abs(xdot_pred.detach()), dim=-1).values,
            "rom_dx_pred_mean": torch.mean(torch.abs(dx_pred.detach()), dim=-1),
            "rom_dx_pred_abs_max": torch.max(torch.abs(dx_pred.detach()), dim=-1).values,
            "rom_pressure_clamp_fraction": torch.mean(pressure_clamp_mask.float(), dim=-1),
            "rom_velocity_clamp_fraction": torch.mean(xdot_clamp_mask.float(), dim=-1),
            "rom_dx_clamp_fraction": torch.mean(dx_clamp_mask.float(), dim=-1),
            "rom_nan_guard_triggered": nan_guard_triggered.float(),
            "rom_output_zeroed_fraction": output_zeroed_fraction,
        }
        return ReducedHydraulicActuatorROMOutput(
            xdot_cyl_pred=xdot_pred,
            dx_cyl_pred=dx_pred,
            P_L_next=P_L_next,
            dP_next=P_L_next,
            valve_cmd_delayed=valve_delayed,
            valve_cmd_effective=valve_effective,
            debug=debug,
        )

    def _param(self, values: Sequence[float], name: str) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.numel() != self.num_cylinders:
            raise ValueError(f"{name} must contain {self.num_cylinders} values, got {tensor.numel()}")
        return tensor.reshape(1, self.num_cylinders)

    def _shape(self, value: torch.Tensor, name: str) -> torch.Tensor:
        value = value.to(device=self.device, dtype=torch.float32)
        if value.shape != (self.num_envs, self.num_cylinders):
            raise ValueError(
                f"{name} must have shape ({self.num_envs}, {self.num_cylinders}), got {tuple(value.shape)}"
            )
        return value

    def _delayed_valve(self, valve_cmd: torch.Tensor) -> torch.Tensor:
        self._delay_buffer = torch.roll(self._delay_buffer, shifts=-1, dims=1)
        self._delay_buffer[:, -1, :] = valve_cmd.detach()
        delayed = torch.empty_like(valve_cmd)
        for cyl_idx in range(self.num_cylinders):
            buffer_index = self.max_delay - int(self.delay_steps[cyl_idx].item())
            delayed[:, cyl_idx] = self._delay_buffer[:, buffer_index, cyl_idx]
        return delayed

    def _deadzone(self, valve_cmd: torch.Tensor) -> torch.Tensor:
        positive = torch.clamp(valve_cmd - self.udz_pos, min=0.0)
        negative = torch.clamp(valve_cmd + self.udz_neg, max=0.0)
        return torch.where(valve_cmd >= 0.0, positive, negative)
