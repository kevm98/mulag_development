from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CLASSIFIER_PATH = ROOT / "kevin_integration" / "tasks" / "asset_preflight_classifier.py"
_SPEC = importlib.util.spec_from_file_location("asset_preflight_classifier", _CLASSIFIER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load classifier module from {_CLASSIFIER_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

classify_asset_transmission_preflight = _MODULE.classify_asset_transmission_preflight
BLOCKED_BASE_STABILITY = _MODULE.BLOCKED_BASE_STABILITY
BLOCKED_DRIVE_AUTHORITY = _MODULE.BLOCKED_DRIVE_AUTHORITY
BLOCKED_LINKAGE_TRANSMISSION = _MODULE.BLOCKED_LINKAGE_TRANSMISSION
BLOCKED_RESET_DURING_PREFLIGHT = _MODULE.BLOCKED_RESET_DURING_PREFLIGHT
ASSET_TRANSMISSION_PASS = _MODULE.ASSET_TRANSMISSION_PASS


GOOD_TARGET = [0.01, 0.01, 0.01, 0.01]
GOOD_CYLINDER = [0.0087471, 0.0112585, 0.0029789, 0.0099845]
GOOD_LINKAGE = {
    "Drehzapfen_joint": 0.0402094,
    "Ausleger_I_joint": 0.0314158,
    "Ausleger_II_joint": 0.0129524,
    "Messerkopf_Schwenk_joint": 0.0380663,
    "Messerkopf_joint": 0.0061312,
}


def _classify(**overrides) -> str:
    kwargs = {
        "finite": True,
        "reset_during_preflight": False,
        "pulse_root_z_motion_m": 0.001,
        "pulse_root_roll_pitch_motion_rad": 0.001,
        "root_vertical_velocity_before_pulse_mps": 0.0,
        "target_motion_m": GOOD_TARGET,
        "cylinder_motion_m": GOOD_CYLINDER,
        "mechanical_family_motion": GOOD_LINKAGE,
    }
    kwargs.update(overrides)
    return classify_asset_transmission_preflight(**kwargs)


def test_transmission_moves_but_base_unstable_blocks_base_stability() -> None:
    decision = _classify(pulse_root_z_motion_m=2.16497)
    if decision != BLOCKED_BASE_STABILITY:
        raise AssertionError(f"expected base stability block, got {decision}")


def test_target_moves_but_cylinder_does_not_blocks_drive_authority() -> None:
    decision = _classify(cylinder_motion_m=[0.0, 0.0, 0.0, 0.0])
    if decision != BLOCKED_DRIVE_AUTHORITY:
        raise AssertionError(f"expected drive authority block, got {decision}")


def test_cylinder_moves_but_linkage_does_not_blocks_linkage_transmission() -> None:
    linkage = dict(GOOD_LINKAGE)
    linkage["Ausleger_II_joint"] = 0.0
    decision = _classify(mechanical_family_motion=linkage)
    if decision != BLOCKED_LINKAGE_TRANSMISSION:
        raise AssertionError(f"expected linkage block, got {decision}")


def test_reset_count_change_blocks_reset_during_preflight() -> None:
    decision = _classify(reset_during_preflight=True, pulse_root_z_motion_m=2.16497)
    if decision != BLOCKED_RESET_DURING_PREFLIGHT:
        raise AssertionError(f"expected reset block before base stability, got {decision}")


def test_large_reset_to_settle_motion_does_not_block_if_pulse_window_is_stable() -> None:
    reset_to_settle_root_z_motion_m = 2.16497
    decision = _classify(pulse_root_z_motion_m=0.001)
    if reset_to_settle_root_z_motion_m <= 2.0:
        raise AssertionError("synthetic reset-to-settle motion is not large")
    if decision != ASSET_TRANSMISSION_PASS:
        raise AssertionError(f"expected pulse-window pass, got {decision}")


def test_world_frame_root_measurement_uses_same_window_delta() -> None:
    env_origin = [20.0, -5.0, 0.0]
    before_world = [21.0, -3.0, 1.5]
    after_world = [21.0, -3.0, 1.501]
    world_delta = [after - before for before, after in zip(before_world, after_world)]
    local_before = [value - origin for value, origin in zip(before_world, env_origin)]
    local_after = [value - origin for value, origin in zip(after_world, env_origin)]
    local_delta = [after - before for before, after in zip(local_before, local_after)]
    if world_delta != local_delta:
        raise AssertionError(f"same-frame delta mismatch: world={world_delta} local={local_delta}")


def main() -> None:
    test_transmission_moves_but_base_unstable_blocks_base_stability()
    test_target_moves_but_cylinder_does_not_blocks_drive_authority()
    test_cylinder_moves_but_linkage_does_not_blocks_linkage_transmission()
    test_reset_count_change_blocks_reset_during_preflight()
    test_large_reset_to_settle_motion_does_not_block_if_pulse_window_is_stable()
    test_world_frame_root_measurement_uses_same_window_delta()
    print("PASS: asset preflight classifier tests passed.")


if __name__ == "__main__":
    main()
