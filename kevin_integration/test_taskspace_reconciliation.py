#!/usr/bin/env python3
"""
Comprehensive test for Kevin task-space hydraulic ROM reconciliation.
This test can run outside of IsaacLab/Conda.
"""

import json
import sys
from pathlib import Path

def test_task_registration() -> dict:
    """Verify task registrations and canonical task setup."""
    result = {"passed": True, "checks": []}

    try:
        import gymnasium as gym
    except ImportError:
        result["passed"] = False
        result["checks"].append({
            "name": "gymnasium_available",
            "passed": False,
            "message": "gymnasium not available (expected when not in IsaacLab)"
        })
        return result

    # Import tasks to register them
    try:
        import kevin_integration.tasks
        result["checks"].append({
            "name": "task_module_import",
            "passed": True,
            "message": "Tasks module imported successfully"
        })
    except Exception as e:
        result["passed"] = False
        result["checks"].append({
            "name": "task_module_import",
            "passed": False,
            "message": f"Failed to import tasks: {e}"
        })
        return result

    # Check canonical task is registered
    canonical_task = "Kevin-Mulag-HydraulicROM-TaskSpace-v0"
    try:
        spec = gym.spec(canonical_task)
        result["checks"].append({
            "name": f"canonical_task_{canonical_task}",
            "passed": True,
            "message": f"Canonical task {canonical_task} is registered"
        })
    except Exception as e:
        result["passed"] = False
        result["checks"].append({
            "name": f"canonical_task_{canonical_task}",
            "passed": False,
            "message": f"Canonical task not registered: {e}"
        })

    # Check aliases
    aliases = [
        "Kevin-Mulag-AdaptiveRL-JointReach-v0",
        "Kevin-Mulag-AdaptiveRL-v0",
        "Kevin-Mulag-HydraulicROM-JointTracking-v0"
    ]
    for alias in aliases:
        try:
            spec = gym.spec(alias)
            result["checks"].append({
                "name": f"alias_{alias}",
                "passed": True,
                "message": f"Alias {alias} is registered"
            })
        except Exception as e:
            result["passed"] = False
            result["checks"].append({
                "name": f"alias_{alias}",
                "passed": False,
                "message": f"Alias not registered: {e}"
            })

    return result


def test_observation_contract() -> dict:
    """Verify task-space observation contract."""
    result = {"passed": True, "checks": []}

    try:
        from kevin_integration.tasks.task_space_reference import (
            TASK_SPACE_OBSERVATION_DIM,
            TASK_SPACE_OBSERVATION_ORDER,
            task_space_observation_dim,
        )

        computed_dim = task_space_observation_dim()
        result["checks"].append({
            "name": "observation_dim_computed",
            "passed": True,
            "message": f"Computed observation dimension: {computed_dim}"
        })

        if computed_dim != TASK_SPACE_OBSERVATION_DIM:
            result["passed"] = False
            result["checks"].append({
                "name": "observation_dim_match",
                "passed": False,
                "message": f"Dimension mismatch: {computed_dim} != {TASK_SPACE_OBSERVATION_DIM}"
            })
        else:
            result["checks"].append({
                "name": "observation_dim_match",
                "passed": True,
                "message": f"Observation dimension verified: {TASK_SPACE_OBSERVATION_DIM}"
            })

        # Verify feature order
        expected_features = 73
        if TASK_SPACE_OBSERVATION_DIM == 73:
            result["checks"].append({
                "name": "observation_features",
                "passed": True,
                "message": f"Task-space features: {len(TASK_SPACE_OBSERVATION_ORDER)} entries, {TASK_SPACE_OBSERVATION_DIM} total dim"
            })
        else:
            result["passed"] = False
            result["checks"].append({
                "name": "observation_features",
                "passed": False,
                "message": f"Unexpected observation dimension: {TASK_SPACE_OBSERVATION_DIM}"
            })

    except Exception as e:
        result["passed"] = False
        result["checks"].append({
            "name": "observation_contract",
            "passed": False,
            "message": f"Failed to verify observation contract: {e}"
        })

    return result


def test_rom_export() -> dict:
    """Verify ROM export package is present and valid."""
    result = {"passed": True, "checks": []}

    try:
        rom_dir = Path(__file__).parent / "rom_cylinders" / "exported_valve_conditioned_causal_rom"

        required_files = [
            "best_valve_conditioned_causal_rom.pth",
            "best_valve_conditioned_causal_rom_scaler.pth",
            "best_valve_conditioned_causal_rom_metadata.json",
        ]

        missing = [f for f in required_files if not (rom_dir / f).is_file()]
        if missing:
            result["passed"] = False
            result["checks"].append({
                "name": "rom_export_files",
                "passed": False,
                "message": f"Missing ROM files: {missing}"
            })
        else:
            result["checks"].append({
                "name": "rom_export_files",
                "passed": True,
                "message": f"All ROM export files present in {rom_dir}"
            })

        # Check metadata
        metadata_file = rom_dir / "best_valve_conditioned_causal_rom_metadata.json"
        if metadata_file.is_file():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

            checks = [
                ("model_version", "valve_conditioned_causal_rom_v1"),
                ("causality_flag", True),
            ]

            for key, expected_value in checks:
                actual = metadata.get(key)
                if actual == expected_value:
                    result["checks"].append({
                        "name": f"rom_metadata_{key}",
                        "passed": True,
                        "message": f"ROM metadata {key}={actual}"
                    })
                else:
                    result["passed"] = False
                    result["checks"].append({
                        "name": f"rom_metadata_{key}",
                        "passed": False,
                        "message": f"ROM metadata mismatch: {key}={actual}, expected {expected_value}"
                    })
    except Exception as e:
        result["passed"] = False
        result["checks"].append({
            "name": "rom_export_check",
            "passed": False,
            "message": f"Failed to verify ROM export: {e}"
        })

    return result


def test_multirate_config() -> dict:
    """Verify multirate ROM configuration."""
    result = {"passed": True, "checks": []}

    try:
        from kevin_integration.tasks.mulag_adaptive_rl_env_cfg import (
            MulagHydraulicROMTaskSpaceEnvCfg,
        )

        cfg = MulagHydraulicROMTaskSpaceEnvCfg()

        # Check multirate constants
        expected_isaac_dt = 0.01
        expected_rom_dt = 0.05
        expected_ratio = 5

        actual_isaac_dt = float(cfg.sim.dt) * float(cfg.decimation)
        actual_rom_dt = float(cfg.final_causal_rom_update_period)
        actual_ratio = int(cfg.final_causal_rom_multirate_ratio)

        # Verify Isaac dt
        if abs(actual_isaac_dt - expected_isaac_dt) < 1e-6:
            result["checks"].append({
                "name": "isaac_dt",
                "passed": True,
                "message": f"Isaac dt: {actual_isaac_dt}s"
            })
        else:
            result["passed"] = False
            result["checks"].append({
                "name": "isaac_dt",
                "passed": False,
                "message": f"Isaac dt mismatch: {actual_isaac_dt} != {expected_isaac_dt}"
            })

        # Verify ROM dt
        if abs(actual_rom_dt - expected_rom_dt) < 1e-6:
            result["checks"].append({
                "name": "rom_dt",
                "passed": True,
                "message": f"ROM dt: {actual_rom_dt}s"
            })
        else:
            result["passed"] = False
            result["checks"].append({
                "name": "rom_dt",
                "passed": False,
                "message": f"ROM dt mismatch: {actual_rom_dt} != {expected_rom_dt}"
            })

        # Verify multirate ratio
        if actual_ratio == expected_ratio:
            result["checks"].append({
                "name": "multirate_ratio",
                "passed": True,
                "message": f"Multirate ratio: {actual_ratio}"
            })
        else:
            result["passed"] = False
            result["checks"].append({
                "name": "multirate_ratio",
                "passed": False,
                "message": f"Multirate ratio mismatch: {actual_ratio} != {expected_ratio}"
            })

    except Exception as e:
        result["passed"] = False
        result["checks"].append({
            "name": "multirate_config",
            "passed": False,
            "message": f"Failed to verify multirate config: {e}"
        })

    return result


def test_control_modes() -> dict:
    """Verify control mode configurations."""
    result = {"passed": True, "checks": []}

    try:
        from kevin_integration.tasks.mulag_adaptive_rl_env_cfg import (
            MulagHydraulicROMTaskSpaceEnvCfg,
            MulagHydraulicROMJointTrackingEnvCfg,
        )

        # Test task-space config
        ts_cfg = MulagHydraulicROMTaskSpaceEnvCfg()
        if ts_cfg.control_mode == "rl_task_space_tracking":
            result["checks"].append({
                "name": "taskspace_control_mode",
                "passed": True,
                "message": f"Task-space config control_mode: {ts_cfg.control_mode}"
            })
        else:
            result["passed"] = False
            result["checks"].append({
                "name": "taskspace_control_mode",
                "passed": False,
                "message": f"Task-space config control_mode wrong: {ts_cfg.control_mode}"
            })

        # Test ROM is enabled
        if ts_cfg.use_final_causal_cylinder_rom:
            result["checks"].append({
                "name": "taskspace_rom_enabled",
                "passed": True,
                "message": "Task-space config has ROM enabled"
            })
        else:
            result["passed"] = False
            result["checks"].append({
                "name": "taskspace_rom_enabled",
                "passed": False,
                "message": "Task-space config ROM not enabled"
            })

        # Test observation space is task-space
        if ts_cfg.observation_space == 73:
            result["checks"].append({
                "name": "taskspace_observation_space",
                "passed": True,
                "message": f"Task-space observation space: {ts_cfg.observation_space}"
            })
        else:
            result["passed"] = False
            result["checks"].append({
                "name": "taskspace_observation_space",
                "passed": False,
                "message": f"Task-space observation space wrong: {ts_cfg.observation_space}"
            })

        # Test joint-tracking config
        jt_cfg = MulagHydraulicROMJointTrackingEnvCfg()
        if jt_cfg.control_mode == "joint_space_validation":
            result["checks"].append({
                "name": "jointtracking_control_mode",
                "passed": True,
                "message": f"Joint-tracking config control_mode: {jt_cfg.control_mode}"
            })
        else:
            result["passed"] = False
            result["checks"].append({
                "name": "jointtracking_control_mode",
                "passed": False,
                "message": f"Joint-tracking config control_mode wrong: {jt_cfg.control_mode}"
            })

    except Exception as e:
        result["passed"] = False
        result["checks"].append({
            "name": "control_modes",
            "passed": False,
            "message": f"Failed to verify control modes: {e}"
        })

    return result


def main() -> int:
    """Run all tests and report results."""
    print("[TEST] Kevin task-space hydraulic ROM reconciliation verification")
    print("=" * 80)

    tests = [
        ("Task Registration", test_task_registration),
        ("Observation Contract", test_observation_contract),
        ("ROM Export", test_rom_export),
        ("Multirate Config", test_multirate_config),
        ("Control Modes", test_control_modes),
    ]

    all_passed = True
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        try:
            result = test_func()
            for check in result.get("checks", []):
                status = "✓" if check["passed"] else "✗"
                print(f"  {status} {check['name']}: {check['message']}")

            if not result.get("passed", False):
                all_passed = False
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("[RESULT] All tests passed!")
        return 0
    else:
        print("[RESULT] Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
