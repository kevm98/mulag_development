"""Safety-shaping contracts for the terrain policy."""

from __future__ import annotations

import torch

from PPO3_SOLE_PhysX.terrain_rewards import (
    bounded_cost_progress,
    compute_verge_reward,
    sanitize_terminal_reward,
)


def _terms(alignment: torch.Tensor, *, scale: float = 0.02):
    n = int(alignment.numel())
    zeros = torch.zeros((n, 4))
    return compute_verge_reward(
        cut_height=torch.full((n,), 0.05),
        target_cut_height_m=0.05,
        lateral_error=torch.zeros(n),
        applied_action=zeros,
        previous_applied_action=zeros,
        head_down_alignment=alignment,
        cutter_contact=torch.zeros(n, dtype=torch.bool),
        cylinder_headroom_low_m=torch.ones((n, 4)),
        cylinder_headroom_high_m=torch.ones((n, 4)),
        tool_acceleration_norm=torch.zeros(n),
        scan_valid_fraction=torch.ones(n),
        head_penalty_weight=-20.0,
        head_threshold=0.995,
        head_penalty_scale=scale,
        head_penalty_max_deviation=2.0,
    )


def test_head_barrier_matches_normalized_quadratic_contract():
    terms = _terms(torch.tensor([0.995, 0.990, 0.980, 0.970]))
    expected = torch.tensor([0.0, -1.25, -11.25, -31.25])
    assert torch.allclose(terms.head_penalty, expected, atol=1.0e-5)


def test_head_barrier_is_zero_above_margin_and_finite_for_zero_scale():
    assert float(_terms(torch.tensor([1.0])).head_penalty[0]) == 0.0
    assert torch.isfinite(_terms(torch.tensor([0.98]), scale=0.0).reward).all()


def test_head_barrier_saturates_before_flipped_states_explode_value_targets():
    terms = _terms(torch.tensor([0.89, 0.0, -1.0]))
    assert torch.equal(terms.head_penalty, torch.full((3,), -80.0))
    assert torch.isfinite(terms.reward).all()


def test_clearance_barrier_warns_before_strike_and_saturates_far_below_ground():
    heights = torch.tensor([0.03, 0.02, 0.01, 0.00, -0.02, -1.00])
    n = int(heights.numel())
    zeros = torch.zeros((n, 4))
    terms = compute_verge_reward(
        cut_height=heights,
        target_cut_height_m=0.05,
        lateral_error=torch.zeros(n),
        applied_action=zeros,
        previous_applied_action=zeros,
        head_down_alignment=torch.ones(n),
        cutter_contact=heights <= 0.0,
        cylinder_headroom_low_m=torch.ones((n, 4)),
        cylinder_headroom_high_m=torch.ones((n, 4)),
        tool_acceleration_norm=torch.zeros(n),
        scan_valid_fraction=torch.ones(n),
        clearance_barrier_weight=-2.0,
        clearance_margin_m=0.02,
        clearance_barrier_scale_m=0.02,
        clearance_barrier_max_deviation=5.0,
    )
    expected = torch.tensor([0.0, 0.0, -0.5, -2.0, -8.0, -50.0])
    assert torch.allclose(terms.clearance_barrier, expected, atol=1.0e-6)


def test_acceleration_penalty_keeps_gradient_through_observed_spikes_and_caps():
    acceleration = torch.tensor([0.0, 30.0, 60.0, 245.0, 300.0, 600.0])
    n = int(acceleration.numel())
    zeros = torch.zeros((n, 4))
    terms = compute_verge_reward(
        cut_height=torch.full((n,), 0.05),
        target_cut_height_m=0.05,
        lateral_error=torch.zeros(n),
        applied_action=zeros,
        previous_applied_action=zeros,
        head_down_alignment=torch.ones(n),
        cutter_contact=torch.zeros(n, dtype=torch.bool),
        cylinder_headroom_low_m=torch.ones((n, 4)),
        cylinder_headroom_high_m=torch.ones((n, 4)),
        tool_acceleration_norm=acceleration,
        scan_valid_fraction=torch.ones(n),
        acceleration_weight=-0.20,
        acceleration_scale_mps2=30.0,
        acceleration_max_ratio=10.0,
    )
    expected = -0.20 * torch.tensor(
        [0.0, 1.0, 4.0, (245.0 / 30.0) ** 2, 100.0, 100.0]
    )
    assert torch.allclose(terms.acceleration_penalty, expected, atol=1.0e-5)
    assert terms.acceleration_penalty[3] < terms.acceleration_penalty[2]
    assert terms.acceleration_penalty[4] == terms.acceleration_penalty[5]


def test_nonfinite_reward_is_replaced_without_changing_healthy_rows():
    reward = torch.tensor([1.25, float("nan"), float("inf"), -3.0])
    sanitized, bad = sanitize_terminal_reward(
        reward,
        invalid_mask=torch.tensor([False, False, False, True]),
        terminal_reward=-100.0,
    )
    assert torch.equal(bad, torch.tensor([False, True, True, True]))
    assert torch.equal(sanitized, torch.tensor([1.25, -100.0, -100.0, -100.0]))
    assert torch.isfinite(sanitized).all()


def test_bounded_cost_progress_has_direction_linear_region_and_hard_cap():
    previous = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    current = torch.tensor([0.5, 0.95, 1.0, 1.05, 2.0])
    progress = bounded_cost_progress(
        previous,
        current,
        scale=0.10,
        weight=0.5,
    )
    assert torch.allclose(
        progress,
        torch.tensor([0.5, 0.25, 0.0, -0.25, -0.5]),
        atol=1.0e-6,
    )


def test_bounded_cost_progress_zero_scale_stays_finite_and_reset_match_is_zero():
    progress = bounded_cost_progress(
        torch.tensor([0.2, 0.2, 0.2]),
        torch.tensor([0.1, 0.2, 0.3]),
        scale=0.0,
        weight=1.0,
    )
    assert torch.equal(progress, torch.tensor([1.0, 0.0, -1.0]))
    assert torch.isfinite(progress).all()


def test_bounded_cost_progress_rejects_shape_mismatch():
    try:
        bounded_cost_progress(
            torch.zeros(2),
            torch.zeros(3),
            scale=1.0,
            weight=1.0,
        )
    except ValueError as exc:
        assert "shapes differ" in str(exc)
    else:
        raise AssertionError("shape mismatch must raise")
