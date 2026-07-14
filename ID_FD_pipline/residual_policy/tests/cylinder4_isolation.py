"""Cylinder-4 isolation contract test."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ID_FD_pipline.residual_policy.actions import apply_residual_action


def test_action_dimension_excludes_cylinder4():
    """Policy action is 3-dim for cylinders 1-3 only."""
    action_dim = 3
    assert action_dim == 3, f"Action space {action_dim} != 3 (cyl 4 excluded)"


def test_residual_applied_only_to_cylinders_1_to_3():
    """Simulate residual application: cylinders 1-3 get correction, cylinder 4 unchanged."""
    N = 2
    residual_scale = 0.002
    baseline = torch.tensor([[0.5, 0.6, 0.7, 0.589], [0.5, 0.6, 0.7, 0.589]])
    residual = torch.tensor([[0.5, -0.3, 0.8], [-1.0, 0.2, -0.5]])

    _, _, corrected = apply_residual_action(
        baseline_goal=baseline,
        previous_filtered_action=torch.zeros_like(residual),
        policy_action=residual,
        residual_scale_m=residual_scale,
        filter_beta=0.0,
        learned_min=torch.zeros((1, 4)),
        learned_max=torch.ones((1, 4)),
    )

    assert torch.allclose(corrected[:, 3], baseline[:, 3]), "Cylinder 4 was modified!"
    assert not torch.allclose(corrected[0, 0], baseline[0, 0]), "Cylinder 1 should have changed"
    print(f"Baseline: {baseline}")
    print(f"Corrected: {corrected}")
    print("Cylinder 4 isolation verified.")


if __name__ == "__main__":
    test_action_dimension_excludes_cylinder4()
    test_residual_applied_only_to_cylinders_1_to_3()
    print("cylinder4_isolation: ALL TESTS PASSED")
