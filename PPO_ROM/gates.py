"""Eight worst-mesh acceptance gates for the final PPO-ROM task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

REQUIRED_MESHES = 3


@dataclass(frozen=True)
class Gate:
    metric: str
    threshold: float
    direction: str  # "max" or "min"
    rationale: str

    def passes(self, value: float) -> bool:
        if self.direction == "max":
            return value <= self.threshold
        if self.direction == "min":
            return value >= self.threshold
        raise ValueError(f"unsupported direction {self.direction!r}")

    def margin(self, value: float) -> float:
        """Signed head-room in metric units. Negative means failing.

        Reported so a pass of 0.0007 cannot masquerade as a comfortable one.
        """
        return (
            self.threshold - value if self.direction == "max" else value - self.threshold
        )


GATES: dict[str, Gate] = {
    "cut_height": Gate(
        "worst_mesh_mean_abs_cut_height_error_m", 0.040, "max",
        "Mowing quality: the cutter must sit within 4 cm of the target height.",
    ),
    "lateral": Gate(
        "worst_mesh_worst_family_mean_abs_lateral_error_m", 0.300, "max",
        "Line following on the WORST path family, not the average one.",
    ),
    "ground_contact": Gate(
        "worst_mesh_worst_family_ground_contact_fraction", 0.700, "min",
        "The cutter must stay on the verge on every family, not on average.",
    ),
    "head_alignment": Gate(
        "worst_mesh_p01_head_alignment", 0.900, "min",
        "Head must stay down; p01 rather than the min, which is one sample.",
    ),
    "hard_impacts": Gate(
        "worst_mesh_hard_impact_step_fraction", 0.350, "max",
        "Impact RATE, not count: a count is not comparable across episode lengths.",
    ),
    "tool_acceleration": Gate(
        "worst_mesh_p95_tool_acceleration_mps2", 120.0, "max",
        "Mechanical wear proxy, at the measured stage-6 floor of ~100.",
    ),
    "cylinder_headroom": Gate(
        "worst_mesh_p01_cylinder_headroom_m", 0.005, "min",
        "The arm must not drive into its own travel limits; p01, not the min.",
    ),
    "safety_envelope": Gate(
        "worst_mesh_safety_violation_rate", 0.080, "max",
        "At most 8% of control steps may violate the raw safety envelope.",
    ),
}


def check_gates(
    per_mesh: Mapping[str, Mapping[str, float]] | Sequence[Mapping[str, float]],
) -> dict:
    """Score a policy from PER-MESH metrics, aggregating to the worst mesh.

    ``per_mesh`` maps a mesh identifier to that mesh's metric dict (or is a
    sequence of them). Raises if fewer than ``REQUIRED_MESHES`` are supplied:
    single-mesh acceptance is the defect this module exists to prevent, so it is
    an error rather than a warning.
    """
    meshes = list(per_mesh.values()) if isinstance(per_mesh, Mapping) else list(per_mesh)
    if len(meshes) < REQUIRED_MESHES:
        raise ValueError(
            f"acceptance needs >= {REQUIRED_MESHES} distinct terrain meshes, got "
            f"{len(meshes)}. Vary the mesh with --eval_terrain_seed; varying "
            "--seed alone returns bit-identical numbers."
        )

    aggregated: dict[str, float] = {}
    for name, gate in GATES.items():
        base = gate.metric.removeprefix("worst_mesh_")
        values = []
        for metrics in meshes:
            if base not in metrics:
                raise KeyError(f"gate {name!r} needs metric {base!r}")
            values.append(float(metrics[base]))
        aggregated[gate.metric] = max(values) if gate.direction == "max" else min(values)

    gates = {n: g.passes(aggregated[g.metric]) for n, g in GATES.items()}
    margins = {n: g.margin(aggregated[g.metric]) for n, g in GATES.items()}
    return {
        "meshes": len(meshes),
        "metrics": aggregated,
        "gates": gates,
        "margins": margins,
        "passed": all(gates.values()),
        "failing": sorted(n for n, ok in gates.items() if not ok),
        "thin_margins": sorted(
            n for n, ok in gates.items()
            if ok and abs(margins[n]) < 0.05 * abs(GATES[n].threshold or 1.0)
        ),
    }
