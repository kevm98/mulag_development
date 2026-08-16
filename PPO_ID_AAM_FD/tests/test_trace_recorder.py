"""The trace is the only evidence for the adaptation figures.

A trace that silently drops rows, misaligns environments, or changes its field
set partway through produces a plot that looks fine and means nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PPO_ID_AAM_FD.trace_recorder import TraceRecorder  # noqa: E402


def test_records_and_saves_with_time_major_shape(tmp_path):
    r = TraceRecorder(num_envs=3, out_path=tmp_path / "t.npz")
    for i in range(5):
        r.record(z_hat=torch.full((3, 2), float(i)),
                 lateral=torch.full((3,), float(i) * 2))
    p = r.save(speed=5.0)
    assert p is not None
    with np.load(p) as d:
        assert d["z_hat"].shape == (5, 3, 2), "expect (steps, envs, width)"
        assert d["lateral"].shape == (5, 3, 1)
        assert int(d["_steps"]) == 5
        assert float(d["meta_speed"]) == 5.0
        # step index must be preserved, not shuffled
        assert np.allclose(d["z_hat"][3], 3.0)


def test_env_dimension_is_not_transposed():
    """Envs must stay on axis 1; a transpose would silently mix plants."""
    r = TraceRecorder(num_envs=4, out_path="unused")
    r.record(x=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    stacked = np.stack(r._cols["x"], axis=0)
    assert stacked.shape == (1, 4, 1)
    assert stacked[0, 2, 0] == 3.0


def test_wrong_env_count_is_rejected():
    r = TraceRecorder(num_envs=4, out_path="unused")
    with pytest.raises(ValueError, match="expected num_envs=4"):
        r.record(x=torch.zeros(3))


def test_field_set_must_stay_constant():
    r = TraceRecorder(num_envs=2, out_path="unused")
    r.record(a=torch.zeros(2), b=torch.zeros(2))
    with pytest.raises(ValueError, match="missing at step"):
        r.record(a=torch.zeros(2))          # dropped b
    with pytest.raises(ValueError, match="appeared at step"):
        r.record(a=torch.zeros(2), b=torch.zeros(2), c=torch.zeros(2))


def test_truncation_is_flagged_not_silent():
    """Hitting max_steps must be visible; a silently short trace would look
    like the episode ended early."""
    r = TraceRecorder(num_envs=1, out_path="unused", max_steps=3)
    for _ in range(6):
        r.record(x=torch.zeros(1))
    assert r.steps == 3
    assert r.truncated is True


def test_save_with_no_rows_returns_none(tmp_path):
    r = TraceRecorder(num_envs=2, out_path=tmp_path / "empty.npz")
    assert r.save() is None
    assert not (tmp_path / "empty.npz").exists()


def test_nonfinite_values_survive_for_diagnosis(tmp_path):
    """NaNs must reach disk rather than being quietly zeroed -- a trace that
    hides them would make a diverged run look healthy."""
    r = TraceRecorder(num_envs=2, out_path=tmp_path / "n.npz")
    r.record(x=torch.tensor([float("nan"), 1.0]))
    p = r.save()
    with np.load(p) as d:
        assert np.isnan(d["x"][0, 0, 0])
        assert d["x"][0, 1, 0] == 1.0


def test_detaches_from_autograd(tmp_path):
    r = TraceRecorder(num_envs=2, out_path=tmp_path / "g.npz")
    x = torch.ones(2, requires_grad=True) * 2
    r.record(x=x)                     # must not raise
    p = r.save()
    with np.load(p) as d:
        assert np.allclose(d["x"], 2.0)
