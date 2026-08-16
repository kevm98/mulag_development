from types import SimpleNamespace

import pytest
import torch

from verge_task.vehicle_speed import (
    advance_path_travel,
    configure_path_horizon,
    episode_path_active_mask,
    fixed_duration_path_limits,
    map_unit_interval_to_speed,
    path_duration_s,
)


def test_path_duration_uses_slowest_episode_speed():
    assert path_duration_s(26.0, 2.0, 0.5) == pytest.approx(47.3)
    assert path_duration_s(26.0, 7.0, 0.5) == pytest.approx(13.8714285714)


def test_configure_path_horizon_preserves_longer_existing_horizon():
    cfg = SimpleNamespace(
        terrain_tracking_active_length_m=26.0,
        vehicle_speed_min_kmh=2.0,
        vehicle_path_completion_margin_s=0.5,
        episode_length_s=60.0,
    )
    assert configure_path_horizon(cfg) == 60.0


def test_fixed_training_segment_replaces_slowest_full_path_horizon():
    cfg = SimpleNamespace(
        terrain_tracking_active_length_m=26.0,
        vehicle_speed_min_kmh=2.0,
        vehicle_path_completion_margin_s=0.5,
        vehicle_training_segment_duration_s=13.4,
        episode_length_s=47.3,
    )
    assert configure_path_horizon(cfg) == pytest.approx(13.9)


def test_path_duration_rejects_nonpositive_speed():
    with pytest.raises(ValueError):
        path_duration_s(26.0, 0.0)


def test_advance_path_travel_clamps_and_latches_completion():
    travel, advance, complete = advance_path_travel(
        torch.tensor([25.99, 12.0]),
        torch.tensor([7.0, 2.0]),
        torch.tensor([True, True]),
        torch.tensor([False, True]),
        path_length_m=26.0,
        dt_s=1.0 / 60.0,
        tolerance_m=1.0e-4,
    )
    assert float(travel[0]) == pytest.approx(26.0)
    assert float(advance[0]) == pytest.approx(0.01, abs=1.0e-5)
    assert bool(complete[0])
    assert float(advance[1]) == pytest.approx(0.0)
    assert not bool(complete[1])


def test_completed_fast_replicas_are_an_absorbing_unscored_tail():
    active = episode_path_active_mask(
        preposition_counter=torch.tensor([0, 0, 1, 0]),
        settle_counter=torch.tensor([0, 0, 0, 1]),
        complete_mask=torch.tensor([False, True, False, False]),
        base_mask=torch.tensor([True, True, True, False]),
    )
    assert active.tolist() == [True, False, False, False]


def test_active_time_balanced_speed_inverse_cdf():
    unit = torch.tensor([0.0, 0.5, 1.0])
    speed = map_unit_interval_to_speed(
        unit, 2.0, 7.0, "active_time_balanced"
    )
    expected = torch.tensor([2.0, (26.5) ** 0.5, 7.0])
    assert torch.allclose(speed, expected)


def test_fixed_speed_is_unchanged_by_sampling_mode():
    unit = torch.tensor([0.0, 0.3, 1.0])
    speed = map_unit_interval_to_speed(
        unit, 4.5, 4.5, "active_time_balanced"
    )
    assert torch.equal(speed, torch.full_like(unit, 4.5))


def test_fixed_duration_segments_finish_together_at_different_distances():
    limits = fixed_duration_path_limits(torch.tensor([2.0, 4.5, 7.0]), 26.0, 13.4)
    expected = torch.tensor([2.0, 4.5, 7.0]) / 3.6 * 13.4
    expected[-1] = 26.0
    assert torch.allclose(limits, expected)

    travel, _, complete = advance_path_travel(
        limits - torch.tensor([0.01, 0.02, 0.03]),
        torch.tensor([2.0, 4.5, 7.0]),
        torch.ones(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.bool),
        path_length_m=limits,
        dt_s=0.1,
        tolerance_m=1.0e-4,
    )
    assert torch.allclose(travel, limits)
    assert complete.tolist() == [True, True, True]
