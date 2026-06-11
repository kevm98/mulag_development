from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RESOLVER_PATH = ROOT / "kevin_integration" / "tasks" / "mechanical_constraint_resolver.py"
_SPEC = importlib.util.spec_from_file_location("mechanical_constraint_resolver", _RESOLVER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load resolver module from {_RESOLVER_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

resolve_mechanical_constraint_family_ids = _MODULE.resolve_mechanical_constraint_family_ids


AVAILABLE_RUNTIME_JOINT_NAMES = (
    "Drehzapfen_constraint_joint_1",
    "Ausleger_I_constraint_joint_3",
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Drehzapfen_constraint_joint_3",
    "Ausleger_I_constraint_joint_1",
    "Ausleger_II_constraint_joint_3",
    "Wheel_Rear_Left_joint",
    "Wheel_Rear_Right_joint",
    "Wheel_Front_Left_Steering_joint",
    "Wheel_Front_Right_Steering_joint",
    "Ausleger_II_joint",
    "Wheel_Front_Left_joint",
    "Wheel_Front_Right_joint",
    "Ausleger_II_constraint_joint_1",
    "Messerkopf_Schwenk_constraint_joint_6",
    "Messerkopf_Schwenk_joint",
    "Messerkopf_Schwenk_constraint_joint_3",
    "Messerkopf_Schwenk_constraint_joint_5",
    "Messerkopf_Schwenk_constraint_joint_1",
    "Messerkopf_joint",
)


def test_resolves_kevin_runtime_mechanical_families() -> None:
    resolution = resolve_mechanical_constraint_family_ids(AVAILABLE_RUNTIME_JOINT_NAMES)
    expected = {
        "Drehzapfen_joint": [0, 4, 2],
        "Ausleger_I_joint": [5, 1, 3],
        "Ausleger_II_joint": [14, 6, 11],
        "Messerkopf_Schwenk_joint": [19, 17, 18, 15, 16],
        "Messerkopf_joint": [20],
    }
    if resolution.family_ids != expected:
        raise AssertionError(f"resolved IDs differ: {resolution.family_ids}")
    if resolution.missing_optional:
        raise AssertionError(f"unexpected missing optional joints: {resolution.missing_optional}")


def test_missing_optional_constraint_joint_warns_without_crashing() -> None:
    names = tuple(name for name in AVAILABLE_RUNTIME_JOINT_NAMES if name != "Messerkopf_Schwenk_constraint_joint_5")
    resolution = resolve_mechanical_constraint_family_ids(names)
    missing = resolution.missing_optional.get("Messerkopf_Schwenk_joint")
    if missing != ["Messerkopf_Schwenk_constraint_joint_5"]:
        raise AssertionError(f"missing optional warning payload differs: {resolution.missing_optional}")
    if "Messerkopf_Schwenk_joint" not in resolution.family_ids:
        raise AssertionError("family was not resolved when an optional constraint was missing")


def test_missing_required_command_joint_is_hard_error() -> None:
    names = tuple(name for name in AVAILABLE_RUNTIME_JOINT_NAMES if name != "Drehzapfen_joint")
    try:
        resolve_mechanical_constraint_family_ids(names)
    except RuntimeError as exc:
        if "Drehzapfen_joint" not in str(exc):
            raise AssertionError(f"hard error did not name the missing required joint: {exc}") from exc
    else:
        raise AssertionError("missing required command joint did not raise")


def main() -> None:
    test_resolves_kevin_runtime_mechanical_families()
    test_missing_optional_constraint_joint_warns_without_crashing()
    test_missing_required_command_joint_is_hard_error()
    print("PASS: mechanical constraint resolver tests passed.")


if __name__ == "__main__":
    main()
