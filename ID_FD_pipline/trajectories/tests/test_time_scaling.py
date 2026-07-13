"""Unit tests for time-scaling functions."""

import math

import torch

from ID_FD_pipline.trajectories.time_scaling import cosine_scaling, quintic_scaling


def test_cosine_endpoints():
    tau = torch.tensor([0.0, 1.0])
    h_p, h_v, h_a = cosine_scaling(tau)
    assert float(h_p[0]) == 0.0, f'h(0)={h_p[0]}'
    assert float(h_p[1]) == 1.0, f'h(1)={h_p[1]}'
    assert abs(float(h_v[0])) < 1e-7, f"h'(0)={h_v[0]}"
    assert abs(float(h_v[1])) < 1e-7, f"h'(1)={h_v[1]}"


def test_cosine_monotonic():
    tau = torch.linspace(0.0, 1.0, 101)
    h_p, _, _ = cosine_scaling(tau)
    diffs = h_p[1:] - h_p[:-1]
    assert (diffs >= -1e-7).all(), f'non-monotonic: min_diff={diffs.min()}'


def test_cosine_symmetry():
    tau = torch.linspace(0.0, 1.0, 51)
    h_p, _, _ = cosine_scaling(tau)
    h_rev, _, _ = cosine_scaling(1.0 - tau)
    max_err = (h_p + h_rev - 1.0).abs().max().item()
    assert max_err < 1e-7, f'symmetry error={max_err}'


def test_cosine_batch():
    tau = torch.rand(100)
    h_p, h_v, h_a = cosine_scaling(tau)
    assert h_p.shape == (100,) and h_v.shape == (100,) and h_a.shape == (100,)
    assert (h_p >= 0.0).all() and (h_p <= 1.0).all()


def test_quintic_endpoints():
    tau = torch.tensor([0.0, 1.0])
    h_p, h_v, h_a = quintic_scaling(tau)
    assert float(h_p[0]) == 0.0 and float(h_p[1]) == 1.0
    assert abs(float(h_v[0])) < 1e-7 and abs(float(h_v[1])) < 1e-7
    assert abs(float(h_a[0])) < 1e-7 and abs(float(h_a[1])) < 1e-7


def test_quintic_monotonic():
    tau = torch.linspace(0.0, 1.0, 101)
    h_p, _, _ = quintic_scaling(tau)
    diffs = h_p[1:] - h_p[:-1]
    assert (diffs >= -1e-7).all()


if __name__ == '__main__':
    test_cosine_endpoints()
    test_cosine_monotonic()
    test_cosine_symmetry()
    test_cosine_batch()
    test_quintic_endpoints()
    test_quintic_monotonic()
    print('[PASS] All time-scaling tests')
