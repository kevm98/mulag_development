"""Layout contract for the sole-controller observation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from PPO3_SOLE_PhysX.observations import (
    IK_HINT_OBS_DIM,
    SOLE_OBS_DIM,
    SOLE_OBS_DIM_WITH_HINT,
    build_sole_observation,
    finite_difference_kinematics,
    sole_observation_dim,
)
from PPO3_SOLE_PhysX.terrain_observations import (
    HEIGHT_PROFILE_BINS,
    build_verge_observation,
    collapse_scan_width,
    nonfinite_row_mask,
    verge_observation_dim,
)


def _signals(n: int = 5, *, fill: float | None = None) -> dict:
    def block(width: int, tag: float) -> torch.Tensor:
        if fill is not None:
            return torch.full((n, width), fill)
        return torch.full((n, width), tag)

    return {
        "tool_error_truck": block(3, 1.0),
        "measured_tool_velocity_truck": block(3, 2.0),
        "reference_tool_velocity_truck": block(3, 3.0),
        "reference_tool_acceleration_truck": block(3, 4.0),
        "measured_cylinder_extensions": block(4, 5.0),
        "measured_cylinder_velocities": block(4, 6.0),
        "cylinder_headroom_low": block(4, 7.0),
        "cylinder_headroom_high": block(4, 8.0),
        "previous_applied_action": block(4, 9.0),
        "current_valve_commands": block(4, 10.0),
        "rom_delta_extension": block(4, 11.0),
        "commanded_target_lead": block(4, 12.0),
        "reference_preview_xy_truck": block(2, 13.0),
        "trajectory_direction": torch.full((n,), 14.0),
        "cutter_bottom_world_clearance": torch.full((n,), 15.0),
        "head_down_alignment": torch.full((n,), 16.0),
        "head_down_alignment_rate": torch.full((n,), 17.0),
    }


def test_default_observation_dimension():
    obs = build_sole_observation(**_signals())
    assert obs.shape == (5, SOLE_OBS_DIM)
    assert sole_observation_dim(ik_hint=False) == SOLE_OBS_DIM


def test_hint_appends_at_the_tail_and_preserves_the_prefix():
    """The ablation must be a strict superset, or the two runs are not comparable."""
    plain = build_sole_observation(**_signals())
    hinted = build_sole_observation(
        **_signals(), ik_hint_cylinder_error=torch.full((5, 4), 99.0)
    )
    assert hinted.shape == (5, SOLE_OBS_DIM_WITH_HINT)
    assert torch.allclose(hinted[:, :SOLE_OBS_DIM], plain)
    assert torch.allclose(hinted[:, SOLE_OBS_DIM:], torch.full((5, 4), 99.0))
    assert SOLE_OBS_DIM_WITH_HINT - SOLE_OBS_DIM == IK_HINT_OBS_DIM


def test_slice_layout_matches_the_documented_contract():
    obs = build_sole_observation(**_signals(n=2))
    expected = [
        (slice(0, 3), 1.0),
        (slice(3, 6), 2.0),
        (slice(6, 9), 3.0),
        (slice(9, 12), 4.0),
        (slice(12, 16), 5.0),
        (slice(16, 20), 6.0),
        (slice(20, 24), 7.0),
        (slice(24, 28), 8.0),
        (slice(28, 32), 9.0),
        (slice(32, 36), 10.0),
        (slice(36, 40), 11.0),
        (slice(40, 44), 12.0),
        (slice(44, 46), 13.0),
        (slice(46, 47), 14.0),
        (slice(47, 48), 15.0),
        (slice(48, 49), 16.0),
        (slice(49, 50), 17.0),
    ]
    for where, value in expected:
        assert torch.allclose(obs[:, where], torch.full_like(obs[:, where], value)), (
            f"slice {where} does not hold the documented signal {value}"
        )


def test_no_cylinder_goal_error_slice_survives_from_ppo2():
    """PPO2's slice 16:20 was the cascade's goal error; here it is velocity.

    Guards against a copy-paste regression that would silently feed the IK
    solution back into the default policy.
    """
    signals = _signals(n=3)
    signals["measured_cylinder_velocities"] = torch.full((3, 4), -1.0)
    obs = build_sole_observation(**signals)
    assert torch.allclose(obs[:, 16:20], torch.full((3, 4), -1.0))


def test_nonfinite_inputs_are_scrubbed():
    signals = _signals()
    signals["tool_error_truck"] = torch.full((5, 3), float("nan"))
    signals["commanded_target_lead"] = torch.full((5, 4), float("inf"))
    obs = build_sole_observation(**signals)
    assert torch.isfinite(obs).all()


def test_raw_nonfinite_mask_preserves_per_environment_fault_location():
    first = torch.zeros((4, 3))
    second = torch.ones((4, 2, 2))
    first[1, 0] = float("nan")
    second[3, 1, 1] = float("inf")
    mask = nonfinite_row_mask(first, second, None)
    assert torch.equal(mask, torch.tensor([False, True, False, True]))


def test_tool_kinematics_first_sample_and_finite_differences():
    position = torch.tensor([[0.0, 0.0, 0.1]])
    velocity, acceleration = finite_difference_kinematics(
        position, None, torch.full_like(position, 99.0), 0.1
    )
    assert torch.equal(velocity, torch.zeros_like(position))
    assert torch.equal(acceleration, torch.zeros_like(position))

    previous_position = torch.zeros_like(position)
    previous_velocity = torch.tensor([[0.0, 0.0, 0.5]])
    velocity, acceleration = finite_difference_kinematics(
        position, previous_position, previous_velocity, 0.1
    )
    assert torch.allclose(velocity, torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(acceleration, torch.tensor([[0.0, 0.0, 5.0]]))

    next_position = torch.tensor([[0.0, 0.0, 0.2]])
    _, next_acceleration = finite_difference_kinematics(
        next_position, position, velocity, 0.1
    )
    assert torch.allclose(next_acceleration, torch.zeros_like(next_position))


def test_verge_scan_collapses_width_with_nearest_row_first():
    # Three lateral rays at each of three along-track locations.
    scan = torch.tensor([[1.0, 2.0, 9.0, 4.0, 5.0, 8.0, 7.0, 6.0, 3.0]])
    profile = collapse_scan_width(scan, bins=3)
    assert torch.equal(profile, torch.tensor([[2.0, 5.0, 6.0]]))


def test_verge_observation_appends_signed_head_axis_after_legacy_prefix():
    n = 3
    four = torch.zeros((n, 4))
    scalar = torch.zeros(n)
    axis_xy = torch.tensor([[0.2, -0.3]]).expand(n, -1)
    obs = build_verge_observation(
        height_profile=torch.zeros((n, HEIGHT_PROFILE_BINS)),
        cut_height_error=scalar,
        lateral_error=scalar,
        cutter_vertical_velocity=scalar,
        measured_cylinder_extensions=four,
        measured_cylinder_velocities=four,
        cylinder_headroom_low=four,
        cylinder_headroom_high=four,
        previous_applied_action=four,
        current_valve_commands=four,
        rom_delta_extension=four,
        commanded_target_lead=four,
        head_down_alignment=scalar,
        head_down_alignment_rate=scalar,
        truck_speed_mps=scalar,
        scan_valid_fraction=torch.ones(n),
        head_down_axis_xy=axis_xy,
        cutter_contact_force_normalized=torch.full((n,), 0.25),
        path_preview=torch.tensor([[0.01, -0.02, 0.03]]).expand(n, -1),
    )
    assert obs.shape == (n, verge_observation_dim() + 1)
    assert obs.shape[-1] == 54
    assert torch.equal(obs[:, :48][:, 47], torch.ones(n))
    assert torch.equal(obs[:, 48:50], axis_xy)
    assert torch.equal(obs[:, 50], torch.full((n,), 0.25))
    assert torch.equal(
        obs[:, 51:54], torch.tensor([[0.01, -0.02, 0.03]]).expand(n, -1)
    )


def test_dimension_mismatch_is_rejected():
    signals = _signals()
    signals["measured_cylinder_extensions"] = torch.zeros((5, 3))
    try:
        build_sole_observation(**signals)
    except RuntimeError:
        return
    raise AssertionError("a malformed observation was accepted")


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
