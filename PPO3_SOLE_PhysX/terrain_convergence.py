"""Shared terrain convergence gates for PPO3 training and evaluation.

The stop condition must mean the same thing in every entry point.  Keeping the
metric-to-threshold mapping here prevents the in-process training diagnostic,
the held-out evaluator, and the persistent supervisor from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class GateSpec:
    """One upper- or lower-bound convergence requirement."""

    metric: str
    threshold: float
    direction: str

    def passes(self, value: float) -> bool:
        if self.direction == "max":
            return value <= self.threshold
        if self.direction == "min":
            return value >= self.threshold
        raise ValueError(f"unsupported gate direction: {self.direction!r}")

    def normalized_violation(self, value: float) -> float:
        """Return zero when passing and relative threshold excess otherwise."""
        scale = max(abs(self.threshold), 1.0e-12)
        if self.direction == "max":
            return max(value / scale - 1.0, 0.0)
        if self.direction == "min":
            return max((self.threshold - value) / scale, 0.0)
        raise ValueError(f"unsupported gate direction: {self.direction!r}")


GATE_SPECS: dict[str, GateSpec] = {
    "mean_cut_height_error": GateSpec(
        "mean_abs_cut_height_error_m", 0.040, "max"
    ),
    "p95_cut_height_error": GateSpec(
        "p95_abs_cut_height_error_m", 0.080, "max"
    ),
    "p95_tool_acceleration": GateSpec(
        # Physical cutter/verge contact produces short solver impulses that the
        # earlier free-space 40 m/s² limit classified as failure even while the
        # truck remained fully supported; 80 m/s² was predeclared for the
        # stage-4 contact task on that basis.
        #
        # Re-derived 2026-07-30 for the stage-5 rough-lawn verge from direct
        # measurement (scripts/probe_tool_acceleration.py, 128 envs x 960
        # held-out steps, reports/tool_accel_zero_floor.json): with the arm
        # commanded to zero the environment alone measures p95 = 100.06 m/s²,
        # so 80 was below the physical floor and unreachable by any policy.
        # The floor's in-contact p95 is 111.6 (the block-1 policy measures an
        # identical 110.4 in contact), and the ground_contact gate requires
        # >= 0.90 contact duty, which bounds any compliant policy's aggregate
        # p95 near ~110: a threshold below that provably contradicts the
        # contact gate. 120 = 1.2 x the measured floor keeps real pressure on
        # the command-attributable share (block-1 policy: 130.9 aggregate,
        # free-swing p95 188.7 vs the floor's 42.8) while staying compatible
        # with mandatory ground contact. Stage-5 scores produced before this
        # change (lawn_rough_stage5_v1-v3 reports) used 80 and are not
        # comparable with scores produced after it.
        "p95_tool_acceleration_mps2", 120.0, "max"
    ),
    # A minimum over 512 x 960 samples becomes stricter merely by adding
    # environments. Gate the lower 1% robustly, while retaining a separate hard
    # minimum that catches an actual head flip/collapse.
    "head_alignment_p01": GateSpec("p01_head_alignment", 0.975, "min"),
    "head_alignment_hard_min": GateSpec("min_head_alignment", 0.950, "min"),
    "lateral_error": GateSpec("mean_abs_lateral_error_m", 0.30, "max"),
    "truck_drift": GateSpec(
        "max_truck_post_constraint_drift_m", 0.001, "max"
    ),
    "truck_heading": GateSpec("max_truck_heading_error_deg", 0.10, "max"),
    # `truck_drift` above is vacuous: `_constrain_truck_xy` assigns
    # `_truck_post_constraint_drift_xy` literal zero, so a 1 mm gate on it can
    # never fail. The sway a viewer actually sees is the chassis error BEFORE
    # the fixture removes it, and until 2026-07-31 nothing measured it.
    #
    # Derived from the zero-action floor exactly as p95_tool_acceleration was
    # (128 x 960 held-out, reports/zero_action_chassis_floor.json): with the arm
    # commanded still the terrain alone produces p95 16.40 mm / 0.202 deg, while
    # the 15/15 policy produces 54.77 mm / 0.629 deg. About 70 % of the sway is
    # therefore command-attributable and removable by training. 1.5 x floor
    # leaves the environment's own contribution comfortably inside the gate
    # while demanding the policy give back most of what it adds.
    "chassis_lateral_reaction": GateSpec(
        "p95_chassis_lateral_reaction_mm", 25.0, "max"
    ),
    "chassis_attitude_reaction": GateSpec(
        "p95_chassis_attitude_reaction_deg", 0.30, "max"
    ),
    "safety_terminations": GateSpec(
        "safety_termination_rate", 0.05, "max"
    ),
    "scan_validity": GateSpec("scan_valid_fraction", 0.80, "min"),
    # Raised 0.90 -> 0.95 on 2026-07-31 to give the convergence supervisor a
    # gradient on cutter skipping, then RESTORED to 0.90 the same day. The
    # gradient was the entire justification and PPO fine-tuning turned out to be
    # unusable here (four campaigns, every block degraded head alignment: p01
    # 0.9854 -> 0.48-0.78 with safety terminations 1.3% -> 3.6-7.7%). 0.95 was a
    # target I chose, not a measured-achievable value -- no configuration ever
    # exceeded 0.9403 -- and leaving a gate nothing can pass is the same mistake
    # as the 80 m/s^2 acceleration gate.
    #
    # This does NOT mean the skipping is solved. The shipped config measures
    # 0.9350, i.e. the cutter leaves the verge ~6.5% of the time, worst family
    # s_curve 0.9193. No physics change and no training run has moved it; see
    # FINAL_REPORT.md for the exhausted list and the free-chassis next step.
    "ground_contact": GateSpec("ground_contact_fraction", 0.90, "min"),
    "wheel_road_support": GateSpec(
        "min_wheel_road_contact_duty_cycle", 0.20, "min"
    ),
    # Aggregate scores can hide one failed path behind ten easy families. These
    # gates require every represented family to retain tracking and contact.
    "worst_family_cut_height": GateSpec(
        "worst_family_mean_abs_cut_height_error_m", 0.050, "max"
    ),
    "worst_family_lateral": GateSpec(
        "worst_family_mean_abs_lateral_error_m", 0.30, "max"
    ),
    "worst_family_contact": GateSpec(
        "worst_family_ground_contact_fraction", 0.85, "min"
    ),
}


def check_gates(metrics: Mapping[str, float]) -> dict[str, bool]:
    """Evaluate every gate, raising clearly when a metric is absent."""
    return {
        name: spec.passes(float(metrics[spec.metric]))
        for name, spec in GATE_SPECS.items()
    }


def normalized_gate_violations(metrics: Mapping[str, float]) -> dict[str, float]:
    """Relative violation per gate; lower is better and zero means pass."""
    return {
        name: spec.normalized_violation(float(metrics[spec.metric]))
        for name, spec in GATE_SPECS.items()
    }


def summarize_convergence(metrics: Mapping[str, float]) -> dict:
    """Build the machine-readable gate result used for checkpoint selection."""
    gates = check_gates(metrics)
    violations = normalized_gate_violations(metrics)
    return {
        "metrics": dict(metrics),
        "gates": gates,
        "normalized_gate_violations": violations,
        "score": float(sum(violations.values())),
        "passed": all(gates.values()),
    }
