"""Unit contracts for the shared PPO3 terrain stop condition."""

from __future__ import annotations

import pytest

from verge_task.terrain_convergence import (
    GATE_SPECS,
    summarize_convergence,
)


def _passing_metrics() -> dict[str, float]:
    return {spec.metric: spec.threshold for spec in GATE_SPECS.values()}


def test_values_exactly_on_every_threshold_pass():
    result = summarize_convergence(_passing_metrics())
    assert result["passed"]
    assert result["score"] == 0.0
    assert all(result["gates"].values())


def test_upper_and_lower_gate_violations_are_normalized():
    metrics = _passing_metrics()
    metrics["mean_abs_cut_height_error_m"] = 0.060
    metrics["p01_head_alignment"] = 0.931

    result = summarize_convergence(metrics)

    assert not result["passed"]
    assert result["normalized_gate_violations"]["mean_cut_height_error"] == pytest.approx(
        0.5
    )
    assert result["normalized_gate_violations"]["head_alignment_p01"] == pytest.approx(
        (GATE_SPECS["head_alignment_p01"].threshold - 0.931)
        / GATE_SPECS["head_alignment_p01"].threshold
    )
    assert result["score"] == pytest.approx(
        0.5
        + (GATE_SPECS["head_alignment_p01"].threshold - 0.931)
        / GATE_SPECS["head_alignment_p01"].threshold
    )


def test_missing_metric_is_rejected_instead_of_silently_passing():
    metrics = _passing_metrics()
    del metrics["ground_contact_fraction"]
    with pytest.raises(KeyError):
        summarize_convergence(metrics)


def test_chassis_reaction_gates_are_derived_from_the_zero_action_floor():
    """The sway gates must sit above what the terrain alone produces.

    p95_tool_acceleration was once gated at 80 against a measured floor of
    100.06 -- a threshold no policy could reach. These two are set at 1.5x the
    measured zero-action floor (16.40 mm / 0.202 deg, 128 x 960 held-out,
    reports/zero_action_chassis_floor.json) so the environment's own
    contribution stays comfortably inside the gate.
    """
    from verge_task.terrain_convergence import GATE_SPECS

    lateral = GATE_SPECS["chassis_lateral_reaction"]
    assert lateral.metric == "p95_chassis_lateral_reaction_mm"
    assert lateral.direction == "max"
    assert lateral.threshold == 25.0
    assert lateral.threshold > 16.40

    attitude = GATE_SPECS["chassis_attitude_reaction"]
    assert attitude.metric == "p95_chassis_attitude_reaction_deg"
    assert attitude.direction == "max"
    assert attitude.threshold == 0.30
    assert attitude.threshold > 0.202


def test_contact_gate_is_not_set_above_any_measured_value():
    """0.95 was briefly used to give the supervisor a gradient, then restored.

    PPO fine-tuning proved unusable here, so the gradient justification is gone,
    and no measured configuration ever exceeded 0.9403. A gate above every
    achievable value is the 80 m/s^2 acceleration mistake repeated.
    """
    from verge_task.terrain_convergence import GATE_SPECS

    assert GATE_SPECS["ground_contact"].threshold == 0.90
    assert GATE_SPECS["ground_contact"].threshold < 0.9403


def test_the_verified_baseline_now_scores_below_a_perfect_sweep():
    """The 15/15 checkpoint must not already satisfy the new objective.

    A supervisor whose best score is 0.0000 has no gradient: `candidate_is_better`
    requires a strictly better score, so every subsequent block would be rejected
    and rolled back regardless of how much sway or skipping it removed.
    """
    from verge_task.terrain_convergence import summarize_convergence

    row = {
        "mean_abs_cut_height_error_m": 0.0076,
        "p95_abs_cut_height_error_m": 0.0498,
        "p95_tool_acceleration_mps2": 111.57,
        "p01_head_alignment": 0.9862,
        "min_head_alignment": 0.9621,
        "mean_abs_lateral_error_m": 0.01,
        "max_truck_post_constraint_drift_m": 0.0,
        "max_truck_heading_error_deg": 0.0,
        "safety_termination_rate": 0.0,
        "scan_valid_fraction": 1.0,
        "ground_contact_fraction": 0.9363,
        "min_wheel_road_contact_duty_cycle": 0.2107,
        "worst_family_mean_abs_cut_height_error_m": 0.0089,
        "worst_family_mean_abs_lateral_error_m": 0.02,
        "worst_family_ground_contact_fraction": 0.9240,
        "p95_chassis_lateral_reaction_mm": 54.77,
        "p95_chassis_attitude_reaction_deg": 0.629,
    }
    summary = summarize_convergence(row)
    failing = {k for k, v in summary["gates"].items() if not v}
    assert failing == {
        "chassis_lateral_reaction",
        "chassis_attitude_reaction",
    }
    assert summary["score"] > 0.0
