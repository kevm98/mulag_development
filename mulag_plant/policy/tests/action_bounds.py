"""Action bounds contract test."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from mulag_plant.policy.actions import apply_residual_action


def test_residual_bounds():
    """Verify residual does not exceed configured scale."""
    scale = 0.002  # 2 mm default
    filter_val = 0.8

    max_raw = torch.tensor([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]])
    baseline = torch.tensor([[0.5, 0.6, 0.7, 0.589]]).expand(2, -1).clone()
    _, _, corrected = apply_residual_action(
        baseline_goal=baseline,
        previous_filtered_action=torch.sign(max_raw),
        policy_action=max_raw,
        residual_scale_m=scale,
        filter_beta=filter_val,
        learned_min=torch.zeros((1, 4)),
        learned_max=torch.ones((1, 4)),
    )
    max_correction = corrected[:, :3] - baseline[:, :3]
    assert (max_correction.abs() <= scale + 1.0e-7).all(), "Correction exceeds scale"
    print(f"Max residual correction: {max_correction}")


def test_clamp_to_learned_limits():
    """Verify corrected cylinder goal is clamped to learned limits."""
    learned_min = torch.tensor([0.0, 0.1, 0.2, 0.3])
    learned_max = torch.tensor([1.0, 1.1, 1.2, 0.8])
    baseline = torch.tensor([0.5, 0.6, 0.7, 0.589])
    residual = torch.tensor([3.0, 3.0, 3.0])
    scale = 0.1
    _, _, corrected = apply_residual_action(
        baseline_goal=baseline.reshape(1, 4),
        previous_filtered_action=torch.zeros((1, 3)),
        policy_action=residual.reshape(1, 3),
        residual_scale_m=scale,
        filter_beta=0.0,
        learned_min=learned_min.reshape(1, 4),
        learned_max=learned_max.reshape(1, 4),
    )
    assert (corrected >= learned_min).all()
    assert (corrected <= learned_max).all()


if __name__ == "__main__":
    test_residual_bounds()
    test_clamp_to_learned_limits()
    print("action_bounds: ALL TESTS PASSED")
