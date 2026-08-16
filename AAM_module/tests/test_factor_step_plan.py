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


def _load_from_collector(*names):
    """Extract the pure function without executing the collector module.

    Importing the collector pulls in Isaac Lab and launches an app, so the
    function is compiled in isolation from source.
    """
    tree = ast.parse(COLLECTOR.read_text())
    ns: dict = {"SystemExit": SystemExit}
    for name in names:
        fn = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(COLLECTOR), "exec"), ns)
    return tuple(ns[n] for n in names)


plan_factor_steps, factor_bounds = _load_from_collector(
    "plan_factor_steps", "factor_bounds"
)
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
    # from_tick/factors are what build_stage31_aam_dataset.load_factor_track
    # consumes; realised_delta is extra provenance and is ignored by it.
    assert {"from_tick", "factors"} <= set(entry)
    assert entry["from_tick"] == 50
    assert entry["factors"] == {"seal": 0.9}
    assert isinstance(entry["from_tick"], int)


def test_step_tick_follows_the_fraction():
    for frac, expected in ((0.25, 250), (0.5, 500), (0.75, 750)):
        plan = plan_factor_steps(ROWS, factor="seal", value=0.9, at_frac=frac,
                                 every_n=1, total_steps=1000)
        assert plan[0]["from_tick"] == expected


def test_exactly_one_of_value_or_delta_is_required():
    with pytest.raises(SystemExit, match="exactly one"):
        plan_factor_steps(ROWS, factor="seal", value=None, at_frac=0.5,
                          every_n=1, total_steps=100)
    with pytest.raises(SystemExit, match="exactly one"):
        plan_factor_steps(ROWS, factor="seal", value=0.9, delta=0.9,
                          at_frac=0.5, every_n=1, total_steps=100)


# --- --step-delta: constant realised magnitude regardless of baseline -------
# Measured on rom_phase2_v1: with --step-value the realised change is
# (value - baseline), which collapsed to 0.027-0.10 for high-baseline runs and
# even went negative. Correlation between realised delta and how well the
# latent tracked was +0.854, so those runs taught the latent to track a change
# that was not in the data.

DRAWS = [{"run_id": 0, "seal": 0.0}, {"run_id": 1, "seal": 0.5},
         {"run_id": 2, "seal": 0.95}, {"run_id": 3, "seal": 1.0}]


def test_delta_realises_the_request_whenever_the_range_allows():
    """0.9 is reachable from 0.0, 0.95 and 1.0, but NOT from 0.5.

    Within bounds of width 1.0 a mid-range baseline simply cannot move 0.9 in
    either direction. That is a property of the factor range, not a defect, so
    the plan realises the largest available move and records it -- letting the
    analysis weight or exclude the short ones instead of silently treating them
    as full-magnitude changes, which is the failure that sank rom_phase2_v1.
    """
    plan = plan_factor_steps(DRAWS, factor="seal", value=None, delta=0.9,
                             at_frac=0.5, every_n=1, total_steps=100,
                             bounds=(0.0, 1.0))
    realised = {r["run_id"]: plan[r["run_id"]]["realised_delta"] for r in DRAWS}
    assert realised[0] == pytest.approx(0.9), "reachable from 0.0"
    assert realised[2] == pytest.approx(0.9), "reachable downward from 0.95"
    assert realised[3] == pytest.approx(0.9), "reachable downward from 1.0"
    # 0.5 is the worst case in a unit range: best possible move is 0.5
    assert realised[1] == pytest.approx(0.5), "mid-range baseline is capped"
    for rid, got in realised.items():
        base = next(r["seal"] for r in DRAWS if r["run_id"] == rid)
        reachable = max(1.0 - base, base - 0.0)
        assert got == pytest.approx(min(0.9, reachable), abs=1e-6)


def test_realised_delta_is_recorded_for_every_stepped_run():
    """Downstream needs it: a short realised delta predicts weak tracking
    (measured correlation +0.854), so it must be inspectable per run."""
    plan = plan_factor_steps(DRAWS, factor="seal", value=None, delta=0.9,
                             at_frac=0.5, every_n=1, total_steps=100,
                             bounds=(0.0, 1.0))
    for entry in plan.values():
        assert "realised_delta" in entry
        assert entry["realised_delta"] >= 0.0


def test_delta_steps_down_when_there_is_no_room_up():
    plan = plan_factor_steps([{"run_id": 9, "seal": 0.95}], factor="seal",
                             value=None, delta=0.9, at_frac=0.5, every_n=1,
                             total_steps=100, bounds=(0.0, 1.0))
    target = plan[9]["factors"]["seal"]
    assert target == pytest.approx(0.05), "must step down, not clamp at the bound"
    assert 0.0 <= target <= 1.0


def test_delta_clamps_when_neither_direction_fits():
    """Delta larger than the whole range: take the larger available move."""
    plan = plan_factor_steps([{"run_id": 9, "seal": 0.4}], factor="seal",
                             value=None, delta=5.0, at_frac=0.5, every_n=1,
                             total_steps=100, bounds=(0.0, 1.0))
    target = plan[9]["factors"]["seal"]
    assert 0.0 <= target <= 1.0
    assert target == pytest.approx(1.0), "0.4 is nearer the low bound, so step up"


def test_value_mode_realises_baseline_dependent_deltas():
    """The documented pitfall, pinned so it cannot be reintroduced silently."""
    plan = plan_factor_steps(DRAWS, factor="seal", value=0.9, at_frac=0.5,
                             every_n=1, total_steps=100, bounds=(0.0, 1.0))
    realised = [plan[r["run_id"]]["realised_delta"] for r in DRAWS]
    assert realised[0] == pytest.approx(0.9)
    assert realised[2] == pytest.approx(0.05, abs=1e-6)
    assert max(realised) - min(realised) > 0.5, (
        "value mode should show baseline-dependent spread; if it does not, the "
        "test fixture no longer exercises the pitfall"
    )


def test_negative_delta_is_rejected():
    with pytest.raises(SystemExit, match="must be positive"):
        plan_factor_steps(DRAWS, factor="seal", value=None, delta=-0.1,
                          at_frac=0.5, every_n=1, total_steps=100)


def test_factor_bounds_come_from_the_manifest():
    rows = [{"seal": 0.0}, {"seal": 0.4}, {"seal": 1.0}]
    assert factor_bounds(rows, "seal") == (0.0, 1.0)
    assert factor_bounds([], "seal") == (0.0, 1.0)


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
