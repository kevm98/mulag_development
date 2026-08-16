from types import SimpleNamespace

import pytest

from PPO_ROM.path_shapes import (
    TRAJECTORY_SHAPES,
    apply_trajectory_shapes,
    parse_trajectory_shapes,
    require_enough_environments,
)


def test_single_shape_selection():
    shapes = parse_trajectory_shapes(single="s_curve")
    cfg = SimpleNamespace(
        tracking_path_family="mixed", tracking_path_family_schedule=()
    )
    apply_trajectory_shapes(cfg, shapes)
    assert cfg.tracking_path_family == "s_curve"
    assert cfg.tracking_path_family_schedule == ()


def test_all_shapes_selection():
    shapes = parse_trajectory_shapes(multiple="all")
    assert shapes == TRAJECTORY_SHAPES
    require_enough_environments(shapes, len(TRAJECTORY_SHAPES))
    with pytest.raises(ValueError, match="num-envs"):
        require_enough_environments(shapes, len(TRAJECTORY_SHAPES) - 1)


def test_subset_preserves_order_and_removes_duplicates():
    shapes = parse_trajectory_shapes(
        multiple="boom_sweep,straight,boom_sweep"
    )
    assert shapes == ("boom_sweep", "straight")


def test_invalid_or_conflicting_selection_is_rejected():
    with pytest.raises(ValueError, match="either"):
        parse_trajectory_shapes(single="straight", multiple="all")
    with pytest.raises(ValueError, match="unknown"):
        parse_trajectory_shapes(single="not_a_shape")
