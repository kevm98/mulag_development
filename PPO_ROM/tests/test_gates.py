"""Contracts for the PPO3_1 gate set. Each test pins a real defect of the old one."""

from __future__ import annotations

import pytest

from PPO_ROM.gates import GATES, REQUIRED_MESHES, check_gates


def _metrics(**over):
    base = {
        "mean_abs_cut_height_error_m": 0.010,
        "worst_family_mean_abs_lateral_error_m": 0.200,
        "worst_family_ground_contact_fraction": 0.920,
        "p01_head_alignment": 0.980,
        "hard_impact_step_fraction": 0.005,
        "p95_tool_acceleration_mps2": 60.0,
        "min_cylinder_headroom_m": 0.030,
        "safety_termination_rate": 0.010,
    }
    base.update(over)
    return base


def test_single_mesh_acceptance_is_refused():
    """The defect that produced a wrong shipping recommendation.

    Every number in the old contract came from pinned mesh 1337. Varying the
    mesh moved min head alignment 0.9507 -> 0.9027 and worst-family lateral
    0.4833 -> 0.3656 -- both the headline pass and the headline failure.
    """
    with pytest.raises(ValueError, match="distinct terrain meshes"):
        check_gates([_metrics()])
    with pytest.raises(ValueError, match="distinct terrain meshes"):
        check_gates({"m1337": _metrics(), "m17": _metrics()})


def test_worst_mesh_decides_not_the_average():
    """A policy that is good on two meshes and bad on one has not passed."""
    result = check_gates({
        "a": _metrics(),
        "b": _metrics(),
        "c": _metrics(worst_family_mean_abs_lateral_error_m=0.480),
    })
    assert result["gates"]["lateral"] is False
    assert result["metrics"]["worst_mesh_worst_family_mean_abs_lateral_error_m"] == 0.480
    assert result["passed"] is False
    assert result["failing"] == ["lateral"]


def test_thin_margins_are_reported_rather_than_rounded_to_true():
    """The old head gate passed at exactly 0.9500 against 0.950 and did not
    generalise. A boolean hid that; a margin does not."""
    result = check_gates({k: _metrics(p01_head_alignment=0.9005) for k in "abc"})
    assert result["gates"]["head_alignment"] is True
    assert "head_alignment" in result["thin_margins"]
    assert result["margins"]["head_alignment"] == pytest.approx(0.0005)


def test_all_gates_pass_on_a_healthy_policy():
    result = check_gates({k: _metrics() for k in "abc"})
    assert result["passed"] is True
    assert result["failing"] == []
    assert result["meshes"] == 3


def test_eight_gates_not_twenty_six():
    """17 standard + 9 strict became 8, each one customer-visible."""
    assert len(GATES) == 8
    assert REQUIRED_MESHES == 3


#: The agreed acceptance contract, pinned value by value.
#:
#: The three re-derived thresholds are marked. Each was re-anchored to the band
#: the privileged DLS teacher actually reaches, because the previous values were
#: above anything this plant has produced -- a gate the ground-truth controller
#: fails measures the threshold, not the policy.
AGREED_CONTRACT = {
    "cut_height": ("max", 0.040),
    "lateral": ("max", 0.300),
    "ground_contact": ("min", 0.700),  # re-derived, was 0.850
    "head_alignment": ("min", 0.900),
    "hard_impacts": ("max", 0.100),  # re-derived, was 0.020
    "tool_acceleration": ("max", 120.0),
    "cylinder_headroom": ("min", 0.005),
    "safety_terminations": ("max", 0.080),  # re-derived, was 0.050
}


def test_thresholds_match_the_agreed_contract_exactly():
    """Pin every threshold, not just the gate count.

    The count was already asserted above and it did not stop the stored
    frontier reports from being scored against superseded thresholds: a gate
    set can drift value-wise while staying eight gates long. Changing any
    number here should require changing this test, which is the point --
    acceptance criteria are a decision, not an implementation detail.
    """
    assert set(GATES) == set(AGREED_CONTRACT)
    actual = {name: (gate.direction, gate.threshold) for name, gate in GATES.items()}
    assert actual == AGREED_CONTRACT


def test_re_derived_thresholds_sit_above_the_measured_teacher_band():
    """The three re-derived gates must admit the DLS teacher's worst mesh.

    Measured in reports/frontier/teacher_gates.json across meshes 1337/17/909.
    If a future edit pushes one of these back above the teacher, the contract
    is once again asking for something no controller here has demonstrated.
    """
    teacher_worst = {
        "ground_contact": 0.7513889074325562,
        "hard_impacts": 0.08704427083333334,
        "safety_terminations": 0.0673828125,
    }
    for name, measured in teacher_worst.items():
        assert GATES[name].passes(measured), (
            f"{name} at {GATES[name].threshold} rejects the privileged teacher's "
            f"measured {measured}"
        )


def test_impact_gate_is_a_rate_so_it_is_comparable_across_episode_lengths():
    """The old set gated a raw COUNT, which also drowned ranking: 1043 impacts
    against a lateral violation of 0.61 meant the supervisor ranked almost
    purely on impacts."""
    assert GATES["hard_impacts"].metric.endswith("_fraction")
    assert GATES["hard_impacts"].threshold < 1.0


def test_every_gate_states_why_it_exists():
    for name, gate in GATES.items():
        assert gate.rationale.strip(), name
        assert gate.direction in ("max", "min"), name


def test_missing_metric_is_an_error_not_a_silent_pass():
    broken = _metrics()
    del broken["p01_head_alignment"]
    with pytest.raises(KeyError, match="p01_head_alignment"):
        check_gates({"a": broken, "b": _metrics(), "c": _metrics()})
