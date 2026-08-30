"""Plant-side perturbations for the AAM online-adaptation experiments.

These change the PLANT, not the observation. The controller never receives the
perturbation parameters; the AAM must infer their effect from the mismatch
between what was commanded and how the system responded.

Applied at the FD boundary::

    u_commanded --> [ PlantPerturbation ] --> u_applied --> FD --> motion
                          alpha, delay

The AAM's feature row carries ``previous_valve`` = the *commanded* valve
(``_prev_valve_cmd``, cloned from ``_current_valve_cmd``). Perturbing only the
applied command therefore leaves the commanded/response mismatch intact, which
is precisely the signal that identifies alpha and the delay. Perturbing the
commanded value instead would hide the effect and invalidate the experiment.

Two perturbations, both chosen because they act directly on the pressure/valve
channel the AAM actually observes:

* ``alpha``  valve effectiveness,  u_applied = alpha * u
* ``delay``  transport delay,      u_applied(t) = u(t - d)

Both support a scheduled mid-episode step change, which is what produces the
adaptation figure: z_hat moves, error spikes, error recovers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PlantPerturbationCfg:
    """Off by default; every field is opt-in."""

    enabled: bool = False

    # valve effectiveness, sampled per environment from [lo, hi]
    alpha_range: tuple[float, float] = (1.0, 1.0)
    # transport delay in CONTROL steps, sampled per environment from [lo, hi]
    delay_steps_range: tuple[int, int] = (0, 0)

    # mid-episode step change. None disables it. At this episode time the
    # per-env alpha is multiplied by ``step_alpha_scale`` and the delay is set
    # to ``step_delay_steps`` (when not None).
    step_time_s: float | None = None
    step_alpha_scale: float = 1.0
    step_delay_steps: int | None = None

    # fixed draw per environment index, so arms are paired env-for-env
    seed: int = 20260816

    # Repeat the per-env draw with this period. The block-44 arm design puts
    # env j and env j+44 on the same terrain so they differ only by arm -- but
    # only if they also get the SAME plant. Without this the two envs draw
    # different alpha/delay and the "paired" difference is contaminated by a
    # plant difference. Set to 44 (11 path families x 4 height scales) to pair
    # exactly; None keeps the original independent draw.
    pair_period: int | None = None

    def validate(self) -> None:
        lo, hi = self.alpha_range
        if not (0.0 < lo <= hi):
            raise ValueError(f"alpha_range must satisfy 0 < lo <= hi, got {self.alpha_range}")
        dlo, dhi = self.delay_steps_range
        if not (0 <= dlo <= dhi):
            raise ValueError(f"delay_steps_range must satisfy 0 <= lo <= hi, got {self.delay_steps_range}")
        if self.step_time_s is not None and self.step_time_s < 0.0:
            raise ValueError("step_time_s must be non-negative")
        if self.step_delay_steps is not None and self.step_delay_steps < 0:
            raise ValueError("step_delay_steps must be non-negative")


class PlantPerturbation:
    """Per-environment valve effectiveness and transport delay."""

    def __init__(self, cfg: PlantPerturbationCfg, *, num_envs: int,
                 device: torch.device | str, control_dt: float,
                 num_cylinders: int = 4) -> None:
        cfg.validate()
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.control_dt = float(control_dt)
        self.num_cylinders = int(num_cylinders)

        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        lo, hi = cfg.alpha_range
        dlo, dhi = cfg.delay_steps_range
        period = cfg.pair_period
        if period is not None:
            period = int(period)
            if period <= 0:
                raise ValueError(f"pair_period must be positive, got {period}")
            if self.num_envs % period:
                raise ValueError(
                    f"pair_period={period} must divide num_envs={self.num_envs} "
                    "so every repeat is complete"
                )
            n_draw = period
        else:
            n_draw = self.num_envs
        alpha_draw = torch.rand(n_draw, generator=gen) * (hi - lo) + lo
        delay_draw = torch.randint(dlo, dhi + 1, (n_draw,), generator=gen)
        if period is not None:
            reps = self.num_envs // period
            alpha_draw = alpha_draw.repeat(reps)
            delay_draw = delay_draw.repeat(reps)
        self.alpha_base = alpha_draw.to(self.device)
        self.delay_base = delay_draw.to(self.device)

        self.alpha = self.alpha_base.clone()
        self.delay = self.delay_base.clone()

        # ring buffer of past APPLIED-stage commands, deep enough for max delay
        self.buffer_len = int(max(int(dhi), int(cfg.step_delay_steps or 0))) + 1
        self.buffer = torch.zeros(
            (self.num_envs, self.buffer_len, self.num_cylinders), device=self.device
        )
        self.write_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self.stepped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (torch.arange(self.num_envs, device=self.device)
               if env_ids is None
               else env_ids.to(self.device, dtype=torch.long).reshape(-1))
        self.buffer[ids] = 0.0
        self.write_idx[ids] = 0
        self.elapsed_s[ids] = 0.0
        self.stepped[ids] = False
        self.alpha[ids] = self.alpha_base[ids]
        self.delay[ids] = self.delay_base[ids]

    @torch.no_grad()
    def apply(self, valve_cmd: torch.Tensor) -> torch.Tensor:
        """Map commanded valve to applied valve. Advances internal time."""
        if not self.enabled:
            return valve_cmd
        u = valve_cmd.to(self.device, dtype=torch.float32)

        # advance episode clock and fire the scheduled step change once
        self.elapsed_s = self.elapsed_s + self.control_dt
        if self.cfg.step_time_s is not None:
            due = (self.elapsed_s >= float(self.cfg.step_time_s)) & ~self.stepped
            if bool(due.any()):
                self.alpha = torch.where(
                    due, self.alpha_base * float(self.cfg.step_alpha_scale), self.alpha
                )
                if self.cfg.step_delay_steps is not None:
                    self.delay = torch.where(
                        due,
                        torch.full_like(self.delay, int(self.cfg.step_delay_steps)),
                        self.delay,
                    )
                self.stepped = self.stepped | due

        # write current command, then read the delayed one
        self.buffer[torch.arange(self.num_envs, device=self.device), self.write_idx] = u
        read_idx = (self.write_idx - self.delay) % self.buffer_len
        delayed = self.buffer[torch.arange(self.num_envs, device=self.device), read_idx]
        self.write_idx = (self.write_idx + 1) % self.buffer_len

        return delayed * self.alpha.reshape(-1, 1)

    def true_theta(self) -> dict[str, torch.Tensor]:
        """Ground truth for LOGGING AND ANALYSIS ONLY.

        Never route this into the policy, the ID model or the AAM input -- the
        whole point is that the controller does not receive theta.
        """
        return {
            "alpha": self.alpha.detach().clone(),
            "delay_steps": self.delay.detach().clone(),
            "stepped": self.stepped.detach().clone(),
            "elapsed_s": self.elapsed_s.detach().clone(),
        }

    def diagnostics(self) -> dict:
        return {
            "enabled": self.enabled,
            "alpha_range": tuple(self.cfg.alpha_range),
            "delay_steps_range": tuple(self.cfg.delay_steps_range),
            "step_time_s": self.cfg.step_time_s,
            "step_alpha_scale": self.cfg.step_alpha_scale,
            "step_delay_steps": self.cfg.step_delay_steps,
            "seed": self.cfg.seed,
            "alpha_mean": float(self.alpha.mean()),
            "delay_mean": float(self.delay.to(torch.float32).mean()),
            "stepped_envs": int(self.stepped.sum()),
        }


# --------------------------------------------------------------------------
# ROM plant
# --------------------------------------------------------------------------
# The command ROM has a DIFFERENT interface from ID-FD. It does not consume a
# valve command -- it consumes a desired cylinder command and PRODUCES the
# valve (`rom_out.predicted_valve`). So valve effectiveness is not applicable
# here. Its native per-env plant channel is `system_factors`, shape
# (num_envs, 7), in the order documented by
# `AAM_module/aam_data/master_schema.e_to_system_factors`:
#
#     [compliance, flow, pressure, seal, seal, seal, seal]
#
# seal occupies four slots (one per cylinder). mass_scale and com_z are NOT in
# this channel; they are mechanical and are applied to the rigid body instead.

SYSTEM_FACTOR_ORDER = ("compliance", "flow", "pressure", "seal", "seal", "seal", "seal")
SYSTEM_FACTOR_INDEX = {
    "compliance": (0,),
    "flow": (1,),
    "pressure": (2,),
    "seal": (3, 4, 5, 6),
}


@dataclass
class RomFactorPerturbationCfg:
    """Per-env ROM hydraulic factors, with an optional mid-episode step.

    Ranges are given per NAMED factor and expanded to the 7-D channel, so the
    caller never has to know that seal occupies four slots.
    """

    enabled: bool = False
    # name -> (lo, hi), sampled per environment
    ranges: dict = field(default_factory=dict)
    # mid-episode step: name -> value applied from step_time_s onwards
    step_time_s: float | None = None
    step_values: dict = field(default_factory=dict)
    seed: int = 20260816

    def validate(self) -> None:
        for name, rng in self.ranges.items():
            if name not in SYSTEM_FACTOR_INDEX:
                raise ValueError(
                    f"unknown ROM factor {name!r}; expected one of "
                    f"{sorted(SYSTEM_FACTOR_INDEX)}"
                )
            lo, hi = rng
            if lo > hi:
                raise ValueError(f"{name} range must satisfy lo <= hi, got {rng}")
        for name in self.step_values:
            if name not in SYSTEM_FACTOR_INDEX:
                raise ValueError(
                    f"unknown ROM factor {name!r} in step_values; expected one of "
                    f"{sorted(SYSTEM_FACTOR_INDEX)}"
                )
        if self.step_time_s is not None and self.step_time_s < 0.0:
            raise ValueError("step_time_s must be non-negative")


class RomFactorPerturbation:
    """Drives `_command_rom_system_factors` for the command-ROM plant."""

    N_FACTORS = 7

    def __init__(self, cfg: RomFactorPerturbationCfg, *, num_envs: int,
                 device: torch.device | str, control_dt: float) -> None:
        cfg.validate()
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.control_dt = float(control_dt)

        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        base = torch.zeros(self.num_envs, self.N_FACTORS)
        for name, (lo, hi) in cfg.ranges.items():
            draw = torch.rand(self.num_envs, generator=gen) * (hi - lo) + lo
            for idx in SYSTEM_FACTOR_INDEX[name]:
                base[:, idx] = draw
        self.base = base.to(self.device)

        self.post = self.base.clone()
        for name, value in cfg.step_values.items():
            for idx in SYSTEM_FACTOR_INDEX[name]:
                self.post[:, idx] = float(value)

        self.factors = self.base.clone()
        self.elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self.stepped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (torch.arange(self.num_envs, device=self.device)
               if env_ids is None
               else env_ids.to(self.device, dtype=torch.long).reshape(-1))
        self.factors[ids] = self.base[ids]
        self.elapsed_s[ids] = 0.0
        self.stepped[ids] = False

    @torch.no_grad()
    def advance(self) -> torch.Tensor | None:
        """Advance the episode clock and return the (num_envs, 7) channel."""
        if not self.enabled:
            return None
        self.elapsed_s = self.elapsed_s + self.control_dt
        if self.cfg.step_time_s is not None and self.cfg.step_values:
            due = (self.elapsed_s >= float(self.cfg.step_time_s)) & ~self.stepped
            if bool(due.any()):
                self.factors = torch.where(due.reshape(-1, 1), self.post, self.factors)
                self.stepped = self.stepped | due
        return self.factors

    def true_theta(self) -> dict[str, torch.Tensor]:
        """Ground truth for LOGGING AND ANALYSIS ONLY."""
        out = {"stepped": self.stepped.detach().clone(),
               "elapsed_s": self.elapsed_s.detach().clone()}
        for name, idxs in SYSTEM_FACTOR_INDEX.items():
            out[f"rom_{name}"] = self.factors[:, idxs[0]].detach().clone()
        return out

    def diagnostics(self) -> dict:
        return {
            "enabled": self.enabled,
            "channel_order": list(SYSTEM_FACTOR_ORDER),
            "ranges": {k: tuple(v) for k, v in self.cfg.ranges.items()},
            "step_time_s": self.cfg.step_time_s,
            "step_values": dict(self.cfg.step_values),
            "seed": self.cfg.seed,
            "stepped_envs": int(self.stepped.sum()),
            "factor_mean": self.factors.mean(dim=0).tolist(),
        }
