"""The perturbation must change the PLANT and stay hidden from the controller.

If any of these fail, the online-adaptation experiment is invalid: either the
plant is not actually changing, or the controller can see the change directly
and no inference from history is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PPO_ID_AAM_FD.plant_perturbation import (  # noqa: E402
    PlantPerturbation, PlantPerturbationCfg,
)


def make(**kw):
    cfg = PlantPerturbationCfg(**kw)
    return PlantPerturbation(cfg, num_envs=4, device="cpu", control_dt=1 / 60)


def test_disabled_is_exactly_identity():
    p = make(enabled=False, alpha_range=(0.5, 0.5), delay_steps_range=(3, 3))
    u = torch.randn(4, 4)
    assert torch.equal(p.apply(u), u), "disabled must not touch the command"


def test_alpha_scales_the_applied_command():
    p = make(enabled=True, alpha_range=(0.7, 0.7))
    u = torch.ones(4, 4)
    assert torch.allclose(p.apply(u), torch.full((4, 4), 0.7))


def test_alpha_is_per_environment_and_deterministic():
    a = make(enabled=True, alpha_range=(0.5, 1.5), seed=7)
    b = make(enabled=True, alpha_range=(0.5, 1.5), seed=7)
    assert torch.allclose(a.alpha, b.alpha), "same seed must pair arms env-for-env"
    assert a.alpha.std() > 0, "alpha must actually vary across envs"
    c = make(enabled=True, alpha_range=(0.5, 1.5), seed=8)
    assert not torch.allclose(a.alpha, c.alpha)


def test_delay_shifts_the_command_in_time():
    p = make(enabled=True, delay_steps_range=(2, 2))
    first = torch.full((4, 4), 5.0)
    # with delay 2 the first two applied commands are the zero-initialised past
    assert torch.allclose(p.apply(first), torch.zeros(4, 4))
    assert torch.allclose(p.apply(torch.full((4, 4), 6.0)), torch.zeros(4, 4))
    # third call returns the first command
    assert torch.allclose(p.apply(torch.full((4, 4), 7.0)), first)


def test_zero_delay_is_passthrough():
    p = make(enabled=True, delay_steps_range=(0, 0))
    u = torch.randn(4, 4)
    assert torch.allclose(p.apply(u), u)


def test_step_change_fires_once_at_the_scheduled_time():
    p = make(enabled=True, alpha_range=(1.0, 1.0),
             step_time_s=0.05, step_alpha_scale=0.5)   # 3 control steps at 60 Hz
    u = torch.ones(4, 4)
    pre = [p.apply(u).clone() for _ in range(2)]
    assert all(torch.allclose(x, torch.ones(4, 4)) for x in pre), "no change before the step"
    post = p.apply(u)
    assert torch.allclose(post, torch.full((4, 4), 0.5)), "alpha must halve at the step"
    assert bool(p.stepped.all())
    # and stay changed, not oscillate
    assert torch.allclose(p.apply(u), torch.full((4, 4), 0.5))


def test_reset_restores_the_baseline_draw():
    p = make(enabled=True, alpha_range=(1.0, 1.0),
             step_time_s=0.0, step_alpha_scale=0.25)
    p.apply(torch.ones(4, 4))
    assert bool(p.stepped.all())
    p.reset()
    assert not bool(p.stepped.any()), "reset must re-arm the step change"
    assert torch.allclose(p.alpha, p.alpha_base), "reset must restore baseline alpha"
    assert torch.allclose(p.buffer, torch.zeros_like(p.buffer))


def test_reset_is_per_environment():
    p = make(enabled=True, alpha_range=(1.0, 1.0),
             step_time_s=0.0, step_alpha_scale=0.25)
    p.apply(torch.ones(4, 4))
    p.reset(torch.tensor([0, 2]))
    assert not bool(p.stepped[0]) and not bool(p.stepped[2])
    assert bool(p.stepped[1]) and bool(p.stepped[3])


def test_true_theta_is_reporting_only_and_matches_state():
    p = make(enabled=True, alpha_range=(0.8, 1.2), delay_steps_range=(1, 3))
    th = p.true_theta()
    assert set(th) == {"alpha", "delay_steps", "stepped", "elapsed_s"}
    assert torch.allclose(th["alpha"], p.alpha)
    # must be a copy: mutating the report must not corrupt the plant
    th["alpha"][:] = 99.0
    assert not torch.allclose(p.alpha, torch.full_like(p.alpha, 99.0))


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="alpha_range"):
        make(enabled=True, alpha_range=(1.5, 0.5))
    with pytest.raises(ValueError, match="alpha_range"):
        make(enabled=True, alpha_range=(0.0, 1.0))
    with pytest.raises(ValueError, match="delay_steps_range"):
        make(enabled=True, delay_steps_range=(3, 1))


def test_buffer_is_deep_enough_for_the_stepped_delay():
    """A step change to a longer delay must not read past the ring buffer."""
    p = make(enabled=True, delay_steps_range=(0, 1),
             step_time_s=0.0, step_delay_steps=5)
    for i in range(12):
        out = p.apply(torch.full((4, 4), float(i)))
        assert torch.isfinite(out).all()
    assert p.buffer_len >= 6
