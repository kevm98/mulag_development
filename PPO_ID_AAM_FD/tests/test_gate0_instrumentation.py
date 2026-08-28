"""Gate 0 asks whether the adaptation site is alive, and needs a positive control.

A dead adaptation site and a robust one look identical from the outside: both
produce a flat curve. E4 spent 33 cells discovering that the deployed latent
was the worst arm without ever measuring whether the residual it carried could
move the valve at all. These tests pin the three failures apart -- the anchor
is on its stops, the residual is clipped away, the residual moves nothing --
and each has a case that must fire, so a silently-zeroed counter cannot pass.

`condition_valve` is exercised against a lightweight stub, so no AAM
checkpoint, no adapter file and no simulator are needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PPO_ID_AAM_FD.online_runtime import OnlineAAMIDFDController  # noqa: E402

LIMIT = 1.0
BOUND = 0.15


class _Stub:
    """Minimal object carrying only what condition_valve touches."""

    def __init__(self, num_envs: int, *, residual: float = 0.05, ready: bool = True):
        self.device = torch.device("cpu")
        self.num_envs = num_envs
        self.enabled = True
        self.aam_ready = torch.full((num_envs,), ready, dtype=torch.bool)
        self.fallback_active = torch.zeros(num_envs, dtype=torch.bool)
        self.z_published = torch.zeros((num_envs, 2))
        self.command_limits = torch.full((1, 4), LIMIT)
        self.residual_bound = torch.full((1, 4), BOUND)
        # A constant-output adapter: tanh(x)*bound is what the real path
        # applies, so feed the inverse to land exactly on `residual`.
        target = max(min(residual / BOUND, 0.999999), -0.999999)
        raw = torch.atanh(torch.tensor(target))
        self.adapter = lambda _x: torch.full((num_envs, 4), float(raw))
        self.adapter_feature_mean = torch.zeros(1)
        self.adapter_feature_std = torch.ones(1)
        self._fallback_count = torch.zeros((), dtype=torch.long)
        for name in (
            "_instr_elements", "_instr_base_at_limit", "_instr_clamp_bound",
            "_instr_residual_dead", "_instr_env_ticks",
            "_instr_fallback_env_ticks", "_instr_applied_env_ticks",
        ):
            setattr(self, name, torch.zeros((), dtype=torch.long))

    def tick(self, base: torch.Tensor):
        n = base.shape[0]
        z = torch.zeros((n, 1))
        return OnlineAAMIDFDController.condition_valve(
            self, u_base=base, extension=z, velocity=z, dP=z,
            force=z, target_dx=z, target_dv=z,
        )

    def read(self):
        return OnlineAAMIDFDController.gate0_instrumentation(self)


def test_healthy_site_reports_a_live_residual():
    """The floor: an unsaturated anchor with a residual that actually moves."""
    s = _Stub(4, residual=0.05)
    s.tick(torch.zeros((4, 4)))
    r = s.read()
    assert r["base_at_limit_fraction"] == 0.0
    assert r["clamp_bound_fraction"] == 0.0
    assert r["residual_dead_fraction"] == 0.0
    assert r["adapter_applied_env_tick_fraction"] == 1.0
    assert r["fallback_env_tick_fraction"] == 0.0


def test_saturated_anchor_is_detected():
    """Positive control for stage 28's 89%: base pinned on its stops.

    The residual is computed, then clipped away, so all three site counters
    must fire together. This is the shape the plan predicts for a bounded
    residual on a saturated anchor.
    """
    s = _Stub(4, residual=0.05)
    s.tick(torch.full((4, 4), LIMIT))
    r = s.read()
    assert r["base_at_limit_fraction"] == 1.0
    assert r["clamp_bound_fraction"] == 1.0
    assert r["residual_dead_fraction"] == 1.0


def test_zero_residual_scale_reads_as_dead_not_as_robust():
    """A residual of zero must not be mistaken for a site that had no effect to have."""
    s = _Stub(4, residual=0.0)
    s.tick(torch.zeros((4, 4)))
    r = s.read()
    assert r["residual_dead_fraction"] == 1.0
    assert r["base_at_limit_fraction"] == 0.0  # the anchor is fine; the residual is not


def test_fallback_ticks_are_counted():
    """0b: ticks that ran on the frozen ID, with no z_hat in the loop at all."""
    s = _Stub(4, residual=0.05)
    s.fallback_active = torch.tensor([True, True, False, False])
    s.tick(torch.zeros((4, 4)))
    r = s.read()
    assert r["fallback_env_tick_fraction"] == 0.5
    assert r["adapter_applied_env_tick_fraction"] == 0.5


def test_counters_accumulate_across_ticks():
    """The readout describes a run, so a single bad tick must not be averaged away."""
    s = _Stub(2, residual=0.05)
    s.tick(torch.zeros((2, 4)))          # healthy
    s.tick(torch.full((2, 4), LIMIT))    # saturated
    r = s.read()
    assert r["valve_elements"] == 16
    assert r["env_ticks"] == 4
    assert r["base_at_limit_fraction"] == 0.5


def test_partial_saturation_is_reported_as_a_fraction():
    """The plan's 0a reading table has three bands; the counter must resolve them."""
    base = torch.zeros((4, 4))
    base[:, 0] = LIMIT      # one of four channels pinned
    s = _Stub(4, residual=0.05)
    s.tick(base)
    r = s.read()
    assert r["base_at_limit_fraction"] == pytest.approx(0.25)
