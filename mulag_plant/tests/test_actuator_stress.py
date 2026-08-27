"""Nominal-invariance and positive-control tests for the Δx-domain stress chain.

Two properties decide whether this feature is safe to enable:

* **Nominal invariance.** With every field at its default the chain must return
  its input BIT-FOR-BIT, so turning the feature on cannot perturb an unstressed
  run. Exact equality, not a tolerance.
* **Positive control.** Each axis, when switched on, must demonstrably change
  the output, and the change must match what was requested. An axis that fails
  this cannot support a robustness claim.

Run:  python3.11 -m pytest mulag_plant/tests/test_actuator_stress.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mulag_plant.runtime.actuator_stress import ActuatorStress, ActuatorStressChain

DT = 0.05
N_ENVS = 6


def sequence(n: int = 60, seed: int = 0) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    # ~1 mm increments, the scale the deployment actually produces
    return [torch.randn(N_ENVS, 4) * 1.0e-3 for _ in range(n)]


def chain(stress: ActuatorStress) -> ActuatorStressChain:
    return ActuatorStressChain(stress, dt_s=DT, num_envs=N_ENVS, max_delay_steps=8)


def test_nominal_is_bitwise_identity():
    c = chain(ActuatorStress())
    for x in sequence():
        assert torch.equal(c.step(x), x)


@pytest.mark.parametrize("stress", [
    ActuatorStress(authority=1.0),
    ActuatorStress(lag_tau_s=0.0),
    ActuatorStress(delay_steps=0),
    ActuatorStress(deadband_m=0.0),
    ActuatorStress(rate_limit_m=float("inf")),
])
def test_each_axis_switched_off_is_identity(stress):
    """Every axis at its nominal setting is an exact identity on its own."""
    c = chain(stress)
    for x in sequence():
        assert torch.equal(c.step(x), x)


def test_authority_scales_exactly():
    c = chain(ActuatorStress(authority=0.4))
    for x in sequence():
        assert torch.allclose(c.step(x), x * 0.4, atol=0.0, rtol=1e-6)


def test_deadband_zeroes_small_increments_and_shrinks_the_rest():
    d = 5.0e-4
    c = chain(ActuatorStress(deadband_m=d))
    for x in sequence():
        out = c.step(x)
        small = x.abs() <= d
        assert torch.equal(out[small], torch.zeros_like(out[small]))
        assert torch.allclose(out[~small].abs(), x[~small].abs() - d, atol=1e-9)


def test_delay_shifts_by_exactly_k_steps():
    k = 3
    c = chain(ActuatorStress(delay_steps=k))
    seq = sequence(20)
    out = [c.step(x) for x in seq]
    for t in range(k, len(seq)):
        assert torch.equal(out[t], seq[t - k])
    for t in range(k):
        assert torch.equal(out[t], torch.zeros_like(seq[0]))


def test_lag_matches_the_closed_form_first_order_response():
    tau = 0.10
    c = chain(ActuatorStress(lag_tau_s=tau))
    beta = 1.0 - math.exp(-DT / tau)
    state = torch.zeros(N_ENVS, 4)
    for x in sequence(30):
        state = state + beta * (x - state)
        assert torch.allclose(c.step(x), state, atol=1e-7)


def test_rate_limit_bounds_the_step_change():
    r = 2.0e-4
    c = chain(ActuatorStress(rate_limit_m=r))
    previous = torch.zeros(N_ENVS, 4)
    for x in sequence():
        out = c.step(x)
        assert (out - previous).abs().max() <= r + 1e-9
        previous = out


def smooth_sequence(n: int = 40) -> list[torch.Tensor]:
    """Phase-shifted sinusoids at ~1 mm, the scale and smoothness of real motion.

    Monotonicity in DELAY cannot be measured on white noise: shifting an i.i.d.
    sequence by any k >= 1 decorrelates it completely, so the deviation from
    nominal saturates at k = 1 and every larger delay ties. A correlated signal
    is required for the delay ladder to be meaningful, and it is also what the
    deployment actually produces.
    """
    out = []
    for t in range(n):
        row = torch.tensor([
            1.0e-3 * math.sin(2.0 * math.pi * t / 20.0 + c * math.pi / 4.0)
            for c in range(4)
        ])
        out.append(row.repeat(N_ENVS, 1))
    return out


def test_severity_is_monotone_in_delivered_effect():
    """More stress must move the output further from nominal, on every axis."""
    seq = smooth_sequence(40)
    baseline = torch.stack([chain(ActuatorStress()).step(x) for x in seq])

    def deviation(stress: ActuatorStress) -> float:
        c = chain(stress)
        out = torch.stack([c.step(x) for x in seq])
        return float((out - baseline).abs().mean())

    ladders = {
        "authority": [ActuatorStress(authority=a) for a in (1.0, 0.8, 0.5, 0.2)],
        "deadband": [ActuatorStress(deadband_m=d) for d in (0.0, 2e-4, 5e-4, 1e-3)],
        "lag": [ActuatorStress(lag_tau_s=t) for t in (0.0, 0.05, 0.15, 0.40)],
        "delay": [ActuatorStress(delay_steps=k) for k in (0, 1, 3, 6)],
    }
    for name, ladder in ladders.items():
        values = [deviation(s) for s in ladder]
        assert all(b > a for a, b in zip(values, values[1:])), f"{name} not monotone: {values}"


def test_delivered_accounting_is_reported():
    c = chain(ActuatorStress(deadband_m=1.0e-3))
    for x in sequence(20):
        c.step(x)
    diag = c.diagnostics()
    assert diag["delivered_steps"] == 20 * N_ENVS * 4
    assert 0.0 < diag["deadband_zeroed_fraction"] <= 1.0


def test_reset_clears_lag_and_delay_state():
    c = chain(ActuatorStress(lag_tau_s=0.2, delay_steps=2))
    for x in sequence(10):
        c.step(x)
    c.reset()
    zero = torch.zeros(N_ENVS, 4)
    assert torch.equal(c.step(zero), zero)


def test_validate_rejects_out_of_range():
    for bad in (
        ActuatorStress(authority=-0.1),
        ActuatorStress(lag_tau_s=-1.0),
        ActuatorStress(delay_steps=99),
        ActuatorStress(deadband_m=-1e-4),
        ActuatorStress(rate_limit_m=0.0),
    ):
        with pytest.raises(ValueError):
            bad.validate(max_delay_steps=8)


def test_nan_input_is_rejected():
    c = chain(ActuatorStress())
    bad = torch.zeros(N_ENVS, 4)
    bad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        c.step(bad)
