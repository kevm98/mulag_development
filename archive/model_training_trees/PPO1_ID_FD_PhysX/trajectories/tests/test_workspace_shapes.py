from __future__ import annotations

from pathlib import Path

import numpy as np

from PPO1_ID_FD_PhysX.trajectories.workspace_shapes import (
    HEIGHT_VARYING_SHAPES,
    SUPPORTED_SHAPES,
    generate_workspace_trajectory,
)


WORKSPACE = Path(__file__).resolve().parents[2] / "models/head_down_workspace_3d.json"


def test_workspace_shapes_are_reachable_and_exciting() -> None:
    for name in SUPPORTED_SHAPES:
        trajectory = generate_workspace_trajectory(WORKSPACE, name)
        assert trajectory.length_m >= 2.5
        assert np.ptp(trajectory.cylinder_seeds[:, 2]) >= 0.05
        assert np.min(trajectory.head_alignment) >= 0.995
        if name in HEIGHT_VARYING_SHAPES:
            assert np.ptp(trajectory.goals[:, 2]) >= 0.20
            assert np.ptp(trajectory.cylinder_seeds[:, 1]) >= 0.02


def test_circle_is_closed() -> None:
    trajectory = generate_workspace_trajectory(WORKSPACE, "circle")
    assert np.linalg.norm(trajectory.goals[0] - trajectory.goals[-1]) < 1.0e-9
