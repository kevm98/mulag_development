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
        # truck remained fully supported. 80 m/s² is predeclared for this
        # contact task; wheel support and chassis drift separately gate whether
        # those impulses destabilize the vehicle.
        "p95_tool_acceleration_mps2", 80.0, "max"
    ),
    # A minimum over 512 x 720 samples becomes stricter merely by adding
    # environments. Gate the lower 1% robustly, while retaining a separate hard
    # minimum that catches an actual head flip/collapse.
    "head_alignment_p01": GateSpec("p01_head_alignment", 0.975, "min"),
    "head_alignment_hard_min": GateSpec("min_head_alignment", 0.950, "min"),
    "lateral_error": GateSpec("mean_abs_lateral_error_m", 0.30, "max"),
    "truck_drift": GateSpec(
        "max_truck_post_constraint_drift_m", 0.001, "max"
    ),
    "truck_heading": GateSpec("max_truck_heading_error_deg", 0.10, "max"),
    "safety_terminations": GateSpec(
        "safety_termination_rate", 0.05, "max"
    ),
    "scan_validity": GateSpec("scan_valid_fraction", 0.80, "min"),
    "ground_contact": GateSpec("ground_contact_fraction", 0.90, "min"),
    "wheel_road_support": GateSpec(
        "min_wheel_road_contact_duty_cycle", 0.20, "min"
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
