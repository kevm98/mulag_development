#!/usr/bin/env python
"""Shared machinery for the E5 CPU stress screens.

Drives the deployed command ROM through its real deployment bridge
(`CommandCylinderROMRuntime`, the class `mulag_upper_level_env` uses in its
`command_rom` branch) against a perfect-follower plant surrogate, and measures
how far a perturbation moves the ROM's dP / velocity / realised extension.

Used by `rom_bulk_modulus_sweep.py` (Stage 0b, one factor) and
`rom_factor_dependence.py` (Stage 1, every physical parameter).

Two things every screen built on this must keep:

* **An all-axes probe.** `command_rom_direction_guard` scales its limit by
  |command|, so an axis at zero command has its response forced to exactly
  zero.  A single-axis probe therefore suppresses the measured sensitivity to
  every hydraulic parameter -- by 122x on the beta grid.  `all_axes` is the
  probe that means anything; `staggered` is kept only to quantify the artifact.
* **A delivered-stress check.** The ROM clamps each physical parameter inside
  `forward`, and training drove five of the eight to their rails, so a
  requested perturbation is often not the applied one.  Never report a null
  without reporting what actually arrived.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rom_model"))

from mulag_plant.runtime.command_rom_runtime import CommandCylinderROMRuntime  # noqa: E402,F401

DP_ABS_LIMIT_PA = 1.5e8  # command_rom_dp_abs_limit_pa
COMMAND_MAX_RATIO = 3.0  # command_rom_command_max_ratio

# Every clamp `HydraulicROM.forward` applies to a positive physical parameter,
# keyed by its state_dict entry. These bound what any stress screen can deliver.
PARAM_CLAMPS: dict[str, tuple[float, float]] = {
    "log_bulk_modulus": (1.0e7, 3.0e9),
    "log_nominal_volume": (1.0e-4, 5.0e-2),
    "log_flow_gain": (1.0e-9, 1.0e-3),
    "log_leakage": (1.0e-15, 1.0e-7),
    "log_supply_pressure": (1.0e6, 5.0e7),
    "log_effective_area": (1.0e-4, 5.0e-2),
    "log_effective_mass": (10.0, 1.0e7),
    "log_viscous_friction": (1.0, 1.0e7),
}


class OutputTap:
    """Capture each `HydraulicROM.forward` result.

    `CommandROMStepOutput` does not carry the residual heads, but the flat
    extension curve can only be attributed if we can see how much of the
    increment is physics and how much is the learned residual running in
    parallel with it.  A forward hook is the least invasive way to get it.
    """

    def __init__(self, model) -> None:
        self.rows: list = []
        self._handle = model.register_forward_hook(lambda _m, _i, out: self.rows.append(out))

    def clear(self) -> None:
        self.rows.clear()

    def close(self) -> None:
        self._handle.remove()

    def shares(self, dt: float, velocities: torch.Tensor) -> tuple[float, float]:
        """Residual share of the dP increment and of the extension increment."""
        if not self.rows:
            return float("nan"), float("nan")
        dp = torch.stack([r.delta_pressure for r in self.rows])
        dp_res = torch.stack([r.pressure_residual for r in self.rows])
        dx = torch.stack([r.delta_extension for r in self.rows])
        dv = torch.stack([r.delta_velocity for r in self.rows])
        # physics half of delta_extension is 0.5*(v + v_next)*dt
        dx_phys = (velocities + 0.5 * dv) * dt
        share_dp = (dp_res.abs().mean() / dp.abs().mean().clamp_min(1e-12)).item()
        share_dx = ((dx - dx_phys).abs().mean() / dx.abs().mean().clamp_min(1e-18)).item()
        return share_dp, share_dx


def safe_trajectory(ticks: int, device: torch.device) -> torch.Tensor:
    """Deterministic staggered trapezoid command, one cylinder at a time.

    Returns [ticks, 1, 8] = cat(target_delta_x, target_delta_xdot).  Amplitude
    stays inside ROM training authority.  Same shape as the trajectory used by
    `AAM_module/tools/rom_sensitivity.py`, so the two screens are comparable.
    """
    command = torch.zeros(ticks, 1, 8, device=device)
    per_cyl = max(ticks // 4, 1)
    for c in range(4):
        start = c * per_cyl
        delta = 1.0e-3  # 1 mm per 50 ms tick
        xdot = delta / 0.05
        for t in range(start, min(start + per_cyl, ticks)):
            phase = t - start
            if phase < 8:
                value = delta
            elif phase < 12:
                value = 0.0
            elif phase < 20:
                value = -delta
            else:
                value = 0.0
            command[t, 0, c] = value
            command[t, 0, 4 + c] = xdot if value != 0.0 else 0.0
    return command


def all_axes_trajectory(ticks: int, device: torch.device) -> torch.Tensor:
    """Phase-shifted sinusoids, all four cylinders commanded on every tick.

    This is the probe that matters.  `command_rom_direction_guard` sets its
    limit proportional to |command|, so an axis sitting at zero command has its
    learned response forced to exactly zero -- the runtime's own source calls
    this out as the reason a single-axis probe under-reads (0.073 of travel
    against 0.827 for direct drive).  `safe_trajectory` drives one cylinder at
    a time and therefore holds 3 of 4 axes in that dead state, which suppresses
    the guard-on response and, with it, any apparent sensitivity to beta.
    In deployment all four axes carry a command.
    """
    command = torch.zeros(ticks, 1, 8, device=device)
    for c in range(4):
        for t in range(ticks):
            value = 1.0e-3 * math.sin(2.0 * math.pi * (t / 20.0) + c * math.pi / 4.0)
            command[t, 0, c] = value
            command[t, 0, 4 + c] = value / 0.05
    return command


TRAJECTORIES = {
    "all_axes": all_axes_trajectory,   # deployment-like: every axis commanded
    "staggered": safe_trajectory,      # single-axis probe; guard-suppressed
}


def build_runtime(checkpoint: str, device: torch.device):
    """The deployed bridge at the ROM's own 50 ms interval.

    `control_dt_s` equal to the ROM interval makes `steps_per_tick` 1, which
    leaves the direction guard's net authority (3x|command| per ROM tick)
    unchanged while keeping the loop at the model's native rate.
    """
    runtime = CommandCylinderROMRuntime(
        checkpoint=checkpoint, num_envs=1, device=device, control_dt_s=0.05
    )
    runtime.model.eval()
    return runtime


def get_param(model, key: str) -> torch.Tensor:
    """Current per-cylinder value of a log-parameterised physical quantity."""
    return model.state_dict()[key].exp().clone()


def set_param(model, key: str, value: torch.Tensor) -> None:
    """Write a positive physical parameter (pre-clamp) into the model."""
    state = model.state_dict()
    state[key] = torch.log(value.clamp_min(1.0e-30))
    model.load_state_dict(state)


def set_raw(model, key: str, value: torch.Tensor) -> None:
    """Write a parameter that is not log-parameterised (additive shifts)."""
    state = model.state_dict()
    state[key] = value
    model.load_state_dict(state)


def delivered(key: str, value: torch.Tensor) -> torch.Tensor:
    """What `forward` actually uses after its own clamp."""
    lo, hi = PARAM_CLAMPS[key]
    return value.clamp(lo, hi)


def rollout(runtime, command: torch.Tensor, limits: torch.Tensor, *, guard: bool, tap=None) -> dict[str, torch.Tensor]:
    """Deployment-faithful rollout: ROM -> guard -> perfect-follower plant."""
    device = runtime.device
    dt = runtime.sample_time_s
    lower, upper = limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0)
    ratio = COMMAND_MAX_RATIO / float(max(runtime.steps_per_tick, 1))

    runtime.reset()
    if tap is not None:
        tap.clear()
    x = (0.5 * (lower + upper)).to(device)  # start mid-stroke, inside every limit
    x0 = x.clone()
    velocity = torch.zeros(1, 4, device=device)
    dP = torch.zeros(1, 4, device=device)

    traj: dict[str, list[torch.Tensor]] = {
        "dP": [], "velocity": [], "extension": [], "velocity_in": [],
        "delta_raw": [], "delta_applied": [], "limit_clipped": [], "guard_vetoed": [],
    }
    for t in range(command.shape[0]):
        traj["velocity_in"].append(velocity.clone())
        out = runtime.step(
            desired_cylinder_command=command[t],
            extension=x,
            extension_velocity=velocity,
            dP=dP,
        )
        delta_raw = out.delta_extension.clone()
        delta = delta_raw
        vetoed = torch.zeros_like(delta, dtype=torch.bool)
        if guard:
            command_delta = command[t][:, :4]
            same_direction = delta * command_delta > 0.0
            limit = ratio * command_delta.abs()
            delta = torch.where(
                same_direction,
                torch.sign(command_delta) * torch.minimum(delta.abs(), limit),
                torch.zeros_like(delta),
            )
            vetoed = ~same_direction

        dP = torch.clamp(dP + out.delta_dP, -DP_ABS_LIMIT_PA, DP_ABS_LIMIT_PA)
        proposed = x + delta
        x_next = torch.maximum(torch.minimum(proposed, upper), lower)
        traj["limit_clipped"].append(proposed.ne(x_next).float())
        # The plant measures velocity; it is not the ROM's own integrator state.
        velocity = (x_next - x) / dt
        x = x_next

        traj["dP"].append(dP.clone())
        traj["velocity"].append(velocity.clone())
        traj["extension"].append(x.clone())
        traj["delta_raw"].append(delta_raw)
        traj["delta_applied"].append(delta.clone())
        traj["guard_vetoed"].append(vetoed.float())
    stacked = {k: torch.stack(v) for k, v in traj.items()}
    stacked["extension_start"] = x0
    if tap is not None:
        share_dp, share_dx = tap.shares(dt, stacked["velocity_in"])
        stacked["residual_share_dP"] = torch.tensor(share_dp)
        stacked["residual_share_extension"] = torch.tensor(share_dx)
    return stacked


def metrics(nominal: dict[str, torch.Tensor], run: dict[str, torch.Tensor]) -> dict:
    out: dict[str, float] = {}
    for signal in ("dP", "velocity", "extension"):
        nom, per = nominal[signal], run[signal]
        span = (nom - nom[0]).abs().amax().clamp_min(1e-12)
        diff = (per - nom).abs()
        out[f"peak_abs_{signal}"] = diff.amax().item()
        out[f"rms_{signal}"] = diff.pow(2).mean().sqrt().item()
        out[f"rel_peak_{signal}"] = (diff.amax() / span).item()
    out["travel_mm"] = ((run["extension"] - run["extension_start"]).abs().amax() * 1e3).item()
    out["peak_extension_err_mm"] = ((run["extension"] - nominal["extension"]).abs().amax() * 1e3).item()
    out["final_extension_err_mm"] = ((run["extension"][-1] - nominal["extension"][-1]).abs().amax() * 1e3).item()
    # Authority actually delivered to the plant, relative to the raw ROM response.
    raw = run["delta_raw"].abs().sum().clamp_min(1e-18)
    out["authority_kept"] = (run["delta_applied"].abs().sum() / raw).item()
    out["guard_veto_rate"] = run["guard_vetoed"].mean().item()
    out["limit_clip_rate"] = run["limit_clipped"].mean().item()
    out["dP_saturation"] = (run["dP"].abs().amax() / DP_ABS_LIMIT_PA).item()
    for key in ("residual_share_dP", "residual_share_extension"):
        if key in run:
            out[key] = run[key].item()
    out["finite"] = bool(torch.isfinite(run["extension"]).all())
    return out
