"""Which runs change plant mid-episode, when, and to what.

`plan_factor_steps` output is written verbatim into each run's
``metadata["factor_schedule"]`` and is then consumed by
``build_stage31_aam_dataset.load_factor_track`` to produce per-window labels.
A wrong plan here is not loud: collection succeeds, training converges, and the
tracking figure is simply measuring nothing.

`plan_factor_steps` is imported without importing the collector's Isaac Lab
dependencies, so these run on CPU with no simulator.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

AAM_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = AAM_ROOT / "tools" / "collect_verge_classical.py"


def _load_plan_factor_steps():
    """Extract the pure function without executing the collector module.

    Importing the collector pulls in Isaac Lab and launches an app, so the
    function is compiled in isolation from source.
    """
    tree = ast.parse(COLLECTOR.read_text())
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "plan_factor_steps"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ns: dict = {"SystemExit": SystemExit}
    exec(compile(module, str(COLLECTOR), "exec"), ns)
    return ns["plan_factor_steps"]


plan_factor_steps = _load_plan_factor_steps()
ROWS = [{"run_id": i} for i in range(6)]


def test_no_factor_means_no_steps():
    """Default collection stays constant-plant, exactly as before."""
    assert plan_factor_steps(ROWS, factor=None, value=None, at_frac=0.5,
                             every_n=2, total_steps=100) == {}


def test_every_n_zero_disables_stepping():
    assert plan_factor_steps(ROWS, factor="seal", value=0.9, at_frac=0.5,
                             every_n=0, total_steps=100) == {}


def test_every_n_selects_a_mixed_subset():
    """A mix keeps between-plant separation learnable AND adds tracking."""
    plan = plan_factor_steps(ROWS, factor="seal", value=0.9, at_frac=0.5,
                             every_n=2, total_steps=100)
    assert sorted(plan) == [0, 2, 4]
    assert len(plan) < len(ROWS), "some runs must stay constant"


def test_every_n_one_steps_all_runs():
    plan = plan_factor_steps(ROWS, factor="seal", value=0.9, at_frac=0.5,
                             every_n=1, total_steps=100)
    assert sorted(plan) == [0, 1, 2, 3, 4, 5]


def test_schedule_shape_matches_the_builder_contract():
    plan = plan_factor_steps(ROWS, factor="seal", value=0.9, at_frac=0.5,
                             every_n=2, total_steps=100)
    entry = plan[0]
    assert set(entry) == {"from_tick", "factors"}
    assert entry["from_tick"] == 50
    assert entry["factors"] == {"seal": 0.9}
    assert isinstance(entry["from_tick"], int)


def test_step_tick_follows_the_fraction():
    for frac, expected in ((0.25, 250), (0.5, 500), (0.75, 750)):
        plan = plan_factor_steps(ROWS, factor="seal", value=0.9, at_frac=frac,
                                 every_n=1, total_steps=1000)
        assert plan[0]["from_tick"] == expected


def test_value_is_required():
    with pytest.raises(SystemExit, match="requires --step-value"):
        plan_factor_steps(ROWS, factor="seal", value=None, at_frac=0.5,
                          every_n=1, total_steps=100)


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_fraction_must_be_strictly_inside_the_episode(frac):
    """A step at tick 0 or at the end is never observed as a CHANGE."""
    with pytest.raises(SystemExit, match="strictly inside"):
        plan_factor_steps(ROWS, factor="seal", value=0.9, at_frac=frac,
                          every_n=1, total_steps=100)


def test_degenerate_tick_is_rejected_on_short_episodes():
    """0.01 of a 10-step episode rounds to tick 0 -- no observable change."""
    with pytest.raises(SystemExit, match="never be observed"):
        plan_factor_steps(ROWS, factor="seal", value=0.9, at_frac=0.01,
                          every_n=1, total_steps=10)


def test_plan_is_keyed_by_run_id_not_batch_position():
    """Slots are reused across batches; keying by position would step the
    wrong runs and mislabel every one of them."""
    rows = [{"run_id": 77}, {"run_id": 12}, {"run_id": 5}]
    plan = plan_factor_steps(rows, factor="seal", value=0.9, at_frac=0.5,
                             every_n=1, total_steps=100)
    assert sorted(plan) == [5, 12, 77]
