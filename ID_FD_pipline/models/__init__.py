"""Checkpoint references for the old cylinder-space ID and FD models."""

from __future__ import annotations

from pathlib import Path


OLD_ID_CHECKPOINT = "id_cylinder_space /runs_id_cylinder/id_cylinder_prevcmd_delta_v2/best_id_cylinder.pt"
OLD_FD_CHECKPOINT = "fd_cylinder_space /runs_fd_cylinder/fd_cylinder_rollout_v2/best_fd_cylinder.pt"

_LOCAL_CYLINDER_MODELS_ROOT = (
    Path(__file__).resolve().parents[2]
    / "kevin_integration"
    / "new_observations_pipeline"
    / "ppo1"
    / "Models_cylinder "
)
_ARCHIVE_CYLINDER_MODELS_ROOT = (
    Path("/home/rrlab/rrlab_mulag_newobs_pruned_20260701")
    / "kevin_integration"
    / "new_observations_pipeline"
    / "Models_cylinder "
)


def _has_old_id_fd(root: Path) -> bool:
    return (root / OLD_ID_CHECKPOINT).is_file() and (root / OLD_FD_CHECKPOINT).is_file()


CYLINDER_MODELS_ROOT = (
    _LOCAL_CYLINDER_MODELS_ROOT
    if _has_old_id_fd(_LOCAL_CYLINDER_MODELS_ROOT)
    else _ARCHIVE_CYLINDER_MODELS_ROOT
)


__all__ = ["CYLINDER_MODELS_ROOT", "OLD_ID_CHECKPOINT", "OLD_FD_CHECKPOINT"]
