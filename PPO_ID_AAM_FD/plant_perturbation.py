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
        self.alpha_base = (
            torch.rand(self.num_envs, generator=gen) * (hi - lo) + lo
        ).to(self.device)
        dlo, dhi = cfg.delay_steps_range
        self.delay_base = torch.randint(
            dlo, dhi + 1, (self.num_envs,), generator=gen
        ).to(self.device)

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
