from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PATH = ROOT / "kevin_integration" / "tasks" / "task_space_reference.py"
_SPEC = importlib.util.spec_from_file_location("task_space_reference", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load task-space reference module from {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_pose_and_quaternion_contract() -> None:
    if _MODULE.TASK_SPACE_POSE_ORDER != ("x", "y", "z", "qw", "qx", "qy", "qz"):
        raise AssertionError(_MODULE.TASK_SPACE_POSE_ORDER)
    if _MODULE.TASK_SPACE_QUATERNION_ORDER != ("qw", "qx", "qy", "qz"):
        raise AssertionError(_MODULE.TASK_SPACE_QUATERNION_ORDER)
    if _MODULE.TASK_SPACE_COMMAND_QUAT_WXYZ != (0.0, 0.0, 1.0, 0.0):
        raise AssertionError(_MODULE.TASK_SPACE_COMMAND_QUAT_WXYZ)


def test_task_space_observation_order_and_dimension() -> None:
    expected = (
        ("ee_pose_command", 7),
        ("current_ee_pose", 7),
        ("ee_position_error", 3),
        ("ee_orientation_error", 1),
        ("mechanical_q", 5),
        ("mechanical_qdot", 5),
        ("cylinder_q", 4),
        ("cylinder_qdot", 4),
        ("previous_action", 4),
        ("previous_ValveCmd", 4),
        ("z_arm_hat", 24),
        ("fd_tau_hat", 4),
        ("vehicle_forward_speed_kmh", 1),
    )
    if _MODULE.TASK_SPACE_OBSERVATION_ORDER != expected:
        raise AssertionError(_MODULE.TASK_SPACE_OBSERVATION_ORDER)
    if _MODULE.task_space_observation_dim() != 73:
        raise AssertionError(_MODULE.task_space_observation_dim())
    _MODULE.validate_task_space_observation_contract()


def test_task_space_command_workspace_matches_rrlab() -> None:
    if _MODULE.task_space_workspace_bounds() != {"x": (6.5, 7.5), "y": (0.7, 2.2), "z": (0.5, 0.5)}:
        raise AssertionError(_MODULE.task_space_workspace_bounds())
    if not _MODULE.position_inside_workspace((6.8, 1.2, 0.5)):
        raise AssertionError("nominal command not inside workspace")


def test_no_ik_bridge_assumption_left_in_active_sources() -> None:
    paths = [
        ROOT / "kevin_integration" / "tasks" / "task_space_reference.py",
        ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py",
        ROOT / "kevin_integration" / "scripts" / "play_mulag_adaptive_rl.py",
        ROOT / "kevin_integration" / "scripts" / "train_mulag_adaptive_rl.py",
    ]
    combined = "\n".join(path.read_text(errors="ignore") for path in paths)
    forbidden = (
        "Differential" + "IKController",
        "BLOCKED_" + "IK_SOLUTION",
        "deterministic " + "IK",
        "body " + "Jacobian",
    )
    found = [term for term in forbidden if term in combined]
    if found:
        raise AssertionError(f"obsolete IK assumption remains: {found}")


def main() -> None:
    test_pose_and_quaternion_contract()
    test_task_space_observation_order_and_dimension()
    test_task_space_command_workspace_matches_rrlab()
    test_no_ik_bridge_assumption_left_in_active_sources()
    print("PASS: task-space reference tests passed.")


if __name__ == "__main__":
    main()
