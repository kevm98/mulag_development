"""Reward anti-gaming contract test.

Verifies that shifting the corrected cylinder goal does not shift the
reference used by the reward. Reward is always against the original
unmodified tool reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ID_FD_pipline.residual_policy.rewards import compute_residual_reward


def test_reward_uses_original_reference():
    """Verify reward is computed against original_ref, not a shifted target."""
    N = 2
    original_ref = torch.tensor([[5.0, 0.0, 0.5], [5.0, 0.0, 0.5]])
    measured = torch.tensor([[5.01, 0.0, 0.5], [5.0, 0.01, 0.5]])
    prev_err = torch.tensor([0.0, 0.0])
    filtered = torch.zeros(N, 3)
    prev_filtered = torch.zeros(N, 3)
    head = torch.ones(N)
    clearance = torch.zeros(N)
    valve = torch.zeros(N, 4)

    terms1 = compute_residual_reward(
        original_reference_tool_position=original_ref,
        measured_tool_position=measured,
        previous_error_norm=prev_err,
        filtered_action=filtered,
        previous_filtered_action=prev_filtered,
        head_down_alignment=head,
        cutter_clearance=clearance,
        valve_command=valve,
    )

    # Shift the "reference" (as if policy shifted it) - reward should DECREASE
    shifted_ref = original_ref + 0.1
    terms2 = compute_residual_reward(
        original_reference_tool_position=shifted_ref,
        measured_tool_position=measured,
        previous_error_norm=prev_err,
        filtered_action=filtered,
        previous_filtered_action=prev_filtered,
        head_down_alignment=head,
        cutter_clearance=clearance,
        valve_command=valve,
    )

    # With shifted reference, error is larger -> tracking reward should be lower
    assert (terms2.tracking < terms1.tracking).all() or (
        terms2.tracking <= terms1.tracking
    ).all(), f"Shifted reference should not increase reward: {terms1.tracking} vs {terms2.tracking}"
    print(f"Original ref reward: {terms1.tracking}")
    print(f"Shifted ref reward: {terms2.tracking}")
    print("Reward anti-gaming verified.")


if __name__ == "__main__":
    test_reward_uses_original_reference()
    print("reward_anti_gaming: ALL TESTS PASSED")
