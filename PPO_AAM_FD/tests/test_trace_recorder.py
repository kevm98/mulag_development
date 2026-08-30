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

from PPO_AAM_FD.trace_recorder import TraceRecorder  # noqa: E402


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


# --------------------------------------------------------------------------
# Where the recorder is CALLED FROM is part of its contract.
#
# The hook lived in `TerrainSolePolicyEnv._get_rewards`, which PPO_ROM's
# `Ppo31SoleEnv` replaces outright without calling super(). Every task in the
# deployment chain inherits from that class, so the hook was dead: traced runs
# wrote a full gates report and no npz, and nothing in the report said so.
# These tests pin the hook to a method no subclass overrides.

import ast as _ast
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
_CHAIN = [
    _REPO / "verge_task/terrain_env.py",
    _REPO / "PPO_ROM/env.py",
    _REPO / "PPO_ID_FD/env.py",
    _REPO / "PPO_AAM_FD/env.py",
]


def _methods_defining(name: str) -> list[_Path]:
    out = []
    for f in _CHAIN:
        tree = _ast.parse(f.read_text())
        if any(isinstance(n, _ast.FunctionDef) and n.name == name
               for n in _ast.walk(tree)):
            out.append(f)
    return out


def _callers_of(call: str) -> set[str]:
    """Names of the methods that call `call`, across the whole chain."""
    found = set()
    for f in _CHAIN:
        tree = _ast.parse(f.read_text())
        for fn in _ast.walk(tree):
            if not isinstance(fn, _ast.FunctionDef):
                continue
            for n in _ast.walk(fn):
                if (isinstance(n, _ast.Call)
                        and isinstance(n.func, _ast.Attribute)
                        and n.func.attr == call):
                    found.add(fn.name)
    return found


def test_the_recorder_is_actually_called():
    assert _callers_of("_record_adaptation_trace"), (
        "nothing calls the recorder; setting adaptation_trace_path would arm "
        "nothing and produce no trace"
    )


def test_the_hook_is_not_on_a_method_a_subclass_replaces():
    """A hook on an overridden method is dead code for the subclass."""
    callers = _callers_of("_record_adaptation_trace")
    for caller in callers:
        definers = _methods_defining(caller)
        assert len(definers) == 1, (
            f"the trace hook sits in {caller!r}, which is defined in "
            f"{[p.name for p in definers]}; a subclass overriding it silences "
            "the recorder without any error"
        )


def test_get_rewards_is_the_negative_control_for_that_rule():
    """Guards the rule above: _get_rewards really is redefined down the chain,
    so the test would catch a hook placed back there."""
    assert len(_methods_defining("_get_rewards")) > 1
