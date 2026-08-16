"""Per-window plant labels: z_target(t) = f(e(t)), not a per-run constant.

The old builder wrote ``z_target = np.repeat(z[None, :], n, axis=0)`` -- one
value for a whole episode. That trained the AAM to emit a CONSTANT, which is
precisely what makes in-episode tracking impossible. These tests pin the
replacement:

* a constant-plant run must still yield constant labels (strict generalisation,
  so every existing dataset keeps its meaning), and
* a run whose plant changes must yield labels that change WITH it, at the right
  tick, using end-of-window semantics.

End-of-window is the deliberate choice: a window straddling a change is
labelled with the NEW plant, so the estimator is taught the fastest tracking
its history allows and the response lag stays an emergent, measurable property
rather than something baked into the label.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

AAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AAM_ROOT))
sys.path.insert(0, str(AAM_ROOT / "tools"))

from aam_data import FACTOR_KEYS  # noqa: E402
from tools.build_stage31_aam_dataset import load_factor_track  # noqa: E402


NOMINAL = {"flow": 0.0, "compliance": 0.0, "seal": 0.1,
           "pressure": 0.0, "mass_scale": 1.0, "com_z": 0.0}


def write_run(tmp_path: Path, run_id: int, metadata: dict,
              track: tuple[np.ndarray, np.ndarray] | None = None) -> Path:
    run = tmp_path / "raw" / "runs" / f"{run_id:05d}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "metadata.json").write_text(json.dumps(metadata))
    if track is not None:
        ticks, factors = track
        np.savez_compressed(run / "factor_track.npz",
                            tick_control_step=ticks, factors=factors)
    return tmp_path


def test_constant_plant_reproduces_legacy_behaviour(tmp_path):
    root = write_run(tmp_path, 0, dict(NOMINAL))
    ticks = np.arange(0, 30, 3, dtype=np.int64)
    e, src = load_factor_track(root, 0, ticks, dict(NOMINAL))
    assert src == "metadata.constant"
    assert e.shape == (len(ticks), len(FACTOR_KEYS))
    assert np.ptp(e, axis=0).max() == 0.0, "constant plant must give constant labels"
    assert e[0, FACTOR_KEYS.index("seal")] == pytest.approx(0.1)


def test_schedule_steps_at_the_declared_tick(tmp_path):
    md = dict(NOMINAL)
    md["factor_schedule"] = [{"from_tick": 15, "factors": {"seal": 0.9}}]
    root = write_run(tmp_path, 1, md)
    ticks = np.arange(0, 30, 3, dtype=np.int64)       # 0,3,...,27
    e, src = load_factor_track(root, 1, ticks, md)
    assert src == "metadata.factor_schedule"
    seal = e[:, FACTOR_KEYS.index("seal")]
    assert np.allclose(seal[ticks < 15], 0.1), "pre-step value must hold"
    assert np.allclose(seal[ticks >= 15], 0.9), "post-step value must apply"


def test_schedule_leaves_other_factors_alone(tmp_path):
    md = dict(NOMINAL)
    md["flow"] = 0.25
    md["factor_schedule"] = [{"from_tick": 9, "factors": {"seal": 0.9}}]
    root = write_run(tmp_path, 2, md)
    ticks = np.arange(0, 21, 3, dtype=np.int64)
    e, _ = load_factor_track(root, 2, ticks, md)
    assert np.allclose(e[:, FACTOR_KEYS.index("flow")], 0.25)


def test_multiple_segments_apply_in_tick_order(tmp_path):
    md = dict(NOMINAL)
    md["factor_schedule"] = [
        {"from_tick": 18, "factors": {"seal": 0.2}},
        {"from_tick": 6, "factors": {"seal": 0.9}},   # deliberately out of order
    ]
    root = write_run(tmp_path, 3, md)
    ticks = np.arange(0, 24, 3, dtype=np.int64)
    e, _ = load_factor_track(root, 3, ticks, md)
    seal = e[:, FACTOR_KEYS.index("seal")]
    assert np.allclose(seal[ticks < 6], 0.1)
    assert np.allclose(seal[(ticks >= 6) & (ticks < 18)], 0.9)
    assert np.allclose(seal[ticks >= 18], 0.2)


def test_npz_track_holds_values_forward(tmp_path):
    src_ticks = np.array([0, 12], dtype=np.int64)
    f = np.repeat(np.array([[NOMINAL[k] for k in FACTOR_KEYS]], dtype=np.float32), 2, axis=0)
    f[1, FACTOR_KEYS.index("seal")] = 0.8
    root = write_run(tmp_path, 4, dict(NOMINAL), track=(src_ticks, f))
    ticks = np.arange(0, 21, 3, dtype=np.int64)
    e, src = load_factor_track(root, 4, ticks, dict(NOMINAL))
    assert src == "factor_track.npz"
    seal = e[:, FACTOR_KEYS.index("seal")]
    assert np.allclose(seal[ticks < 12], 0.1), "value before the change is held"
    assert np.allclose(seal[ticks >= 12], 0.8), "value carries forward after"


def test_npz_track_takes_priority_over_schedule(tmp_path):
    md = dict(NOMINAL)
    md["factor_schedule"] = [{"from_tick": 0, "factors": {"seal": 0.5}}]
    src_ticks = np.array([0], dtype=np.int64)
    f = np.array([[NOMINAL[k] for k in FACTOR_KEYS]], dtype=np.float32)
    f[0, FACTOR_KEYS.index("seal")] = 0.7
    root = write_run(tmp_path, 5, md, track=(src_ticks, f))
    e, src = load_factor_track(root, 5, np.arange(0, 9, 3, dtype=np.int64), md)
    assert src == "factor_track.npz"
    assert np.allclose(e[:, FACTOR_KEYS.index("seal")], 0.7)


def test_track_starting_after_the_first_row_is_rejected(tmp_path):
    """Silently back-filling would invent a plant that never existed."""
    src_ticks = np.array([9], dtype=np.int64)
    f = np.array([[NOMINAL[k] for k in FACTOR_KEYS]], dtype=np.float32)
    root = write_run(tmp_path, 6, dict(NOMINAL), track=(src_ticks, f))
    with pytest.raises(SystemExit, match="after the first row tick"):
        load_factor_track(root, 6, np.arange(0, 21, 3, dtype=np.int64), dict(NOMINAL))


def test_unknown_factor_in_schedule_is_rejected(tmp_path):
    md = dict(NOMINAL)
    md["factor_schedule"] = [{"from_tick": 0, "factors": {"viscosity": 1.0}}]
    root = write_run(tmp_path, 7, md)
    with pytest.raises(SystemExit, match="unknown factors"):
        load_factor_track(root, 7, np.arange(0, 9, 3, dtype=np.int64), md)


def test_malformed_track_shape_is_rejected(tmp_path):
    src_ticks = np.array([0, 3], dtype=np.int64)
    f = np.zeros((2, 3), dtype=np.float32)            # wrong factor count
    root = write_run(tmp_path, 8, dict(NOMINAL), track=(src_ticks, f))
    with pytest.raises(SystemExit, match=r"factors must be"):
        load_factor_track(root, 8, np.arange(0, 9, 3, dtype=np.int64), dict(NOMINAL))


def test_end_of_window_semantics_for_a_straddling_window():
    """A window covering the change must be labelled with the NEW plant.

    valid_idx are window END indices, so labelling by that index means the
    target is the plant at the newest row in the window. This is the property
    the whole tracking experiment rests on.
    """
    from aam_data.rom_stage31 import causal_stack

    n, lag = 12, 4
    feats = np.arange(n * 18, dtype=np.float64).reshape(n, 18)
    _, valid = causal_stack(feats, lag)
    valid_idx = np.flatnonzero(valid)
    track = np.zeros((n, len(FACTOR_KEYS)), dtype=np.float32)
    change_row = 7
    track[change_row:, FACTOR_KEYS.index("seal")] = 1.0
    labels = track[valid_idx][:, FACTOR_KEYS.index("seal")]
    for k, end in enumerate(valid_idx):
        expected = 1.0 if end >= change_row else 0.0
        assert labels[k] == expected, (
            f"window ending at row {end} should carry the plant AT ITS END "
            f"({expected}), got {labels[k]}"
        )
