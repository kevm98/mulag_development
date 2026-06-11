from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CONVENTIONS_PATH = ROOT / "kevin_integration" / "tasks" / "rrlab_reference_conventions.py"
_SPEC = importlib.util.spec_from_file_location("rrlab_reference_conventions", _CONVENTIONS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load conventions module from {_CONVENTIONS_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_arm_and_vehicle_joint_conventions_match_obstacle_task() -> None:
    expected_arm = (
        "Drehzapfen_joint",
        "Ausleger_I_joint",
        "Ausleger_II_joint",
        "Messerkopf_Schwenk_joint",
        "Messerkopf_joint",
    )
    expected_wheels = (
        "Wheel_Front_Left_joint",
        "Wheel_Front_Right_joint",
        "Wheel_Rear_Left_joint",
        "Wheel_Rear_Right_joint",
    )
    if _MODULE.OBSTACLE_ARM_JOINT_NAMES != expected_arm:
        raise AssertionError(_MODULE.OBSTACLE_ARM_JOINT_NAMES)
    if _MODULE.OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES != expected_arm[:4]:
        raise AssertionError(_MODULE.OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES)
    if _MODULE.OBSTACLE_WHEEL_JOINT_NAMES != expected_wheels:
        raise AssertionError(_MODULE.OBSTACLE_WHEEL_JOINT_NAMES)


def test_task_space_reference_constants_match_obstacle_task() -> None:
    if _MODULE.OBSTACLE_TASK_SPACE_BODY_NAME != "Messerkopf":
        raise AssertionError(_MODULE.OBSTACLE_TASK_SPACE_BODY_NAME)
    if _MODULE.OBSTACLE_TASK_SPACE_COMMAND_NAME != "ee_pose":
        raise AssertionError(_MODULE.OBSTACLE_TASK_SPACE_COMMAND_NAME)
    if _MODULE.OBSTACLE_TASK_SPACE_POS_X_RANGE_M != (6.5, 7.5):
        raise AssertionError(_MODULE.OBSTACLE_TASK_SPACE_POS_X_RANGE_M)
    if _MODULE.OBSTACLE_TASK_SPACE_POS_Y_RANGE_M != (0.7, 2.2):
        raise AssertionError(_MODULE.OBSTACLE_TASK_SPACE_POS_Y_RANGE_M)
    if _MODULE.OBSTACLE_TASK_SPACE_POS_Z_RANGE_M != (0.5, 0.5):
        raise AssertionError(_MODULE.OBSTACLE_TASK_SPACE_POS_Z_RANGE_M)
    if _MODULE.OBSTACLE_TASK_SPACE_RESAMPLING_TIME_RANGE_S != (10.0, 12.0):
        raise AssertionError(_MODULE.OBSTACLE_TASK_SPACE_RESAMPLING_TIME_RANGE_S)


def test_vehicle_speed_conversion_matches_rrlab_randomizer() -> None:
    lower, upper = _MODULE.obstacle_speed_range_rad_per_sec()
    if not math.isclose(lower, (3.0 / 3.6) / 0.575, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError(lower)
    if not math.isclose(upper, (8.0 / 3.6) / 0.575, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError(upper)
    if not math.isclose(_MODULE.wheel_rad_per_sec_to_kmh(lower), 3.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError(lower)


def test_kevin_source_keeps_vehicle_command_separate_from_rom_action_contract() -> None:
    cfg_source = (ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env_cfg.py").read_text()
    env_source = (ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py").read_text()
    if "action_space = 4" not in cfg_source:
        raise AssertionError("Kevin hydraulic ROM action_space changed from 4")
    if "observation_space = 62" not in cfg_source:
        raise AssertionError("Kevin hydraulic ROM observation_space changed from 62")
    if "task_space_observation_space = TASK_SPACE_OBSERVATION_DIM" not in cfg_source:
        raise AssertionError("Kevin task-space observation dimension is not sourced from the contract")
    if "vehicle_command_mode = \"rrlab_obstacle_scripted_wheel_velocity\"" not in cfg_source:
        raise AssertionError("Kevin cfg does not select the RRLAB scripted wheel velocity mode")
    for needle in (
        "enable_scripted_vehicle_motion = True",
        "vehicle_speed_min_kmh = OBSTACLE_DRIVING_SPEED_RANGE_KMH[0]",
        "vehicle_speed_max_kmh = OBSTACLE_DRIVING_SPEED_RANGE_KMH[1]",
    ):
        if needle not in cfg_source:
            raise AssertionError(f"missing vehicle config field: {needle}")
    if "reference_marker_debug_vis = False" not in cfg_source:
        raise AssertionError("Synthetic joint-reference marker should stay disabled by default")
    if "self.robot.write_joint_velocity_to_sim" not in env_source:
        raise AssertionError("Kevin env does not write wheel velocity targets")
    if "resample_mask = self._vehicle_commanded_speed_kmh <= 0.0" not in env_source:
        raise AssertionError("Kevin vehicle speed should be sampled once after reset, not periodically")
    if "self._apply_rrlab_vehicle_command()" not in env_source:
        raise AssertionError("Kevin env does not apply the vehicle command from _apply_action")
    for needle in (
        "torch.clamp(q_ref, safe_lower, safe_upper)",
        "self._apply_joint5_coupling_to_reference(q_ref",
        "build_rl_obs(",
        "joint_reach_reward_debug(",
        "self._current_qddot_cmd = self._controller.build_qddot_cmd(",
        "self.robot.set_joint_position_target(self._rom_cylinder_pos_target",
    ):
        if needle not in env_source:
            raise AssertionError(f"missing Kevin action/reference/reward source evidence: {needle}")


def test_rrlab_obstacle_randomizer_source_uses_scripted_wheel_velocity() -> None:
    source = (
        ROOT
        / "extensions"
        / "rrlab_tasks"
        / "rrlab_tasks"
        / "manager_based"
        / "manipulation"
        / "obstacle_avoidance"
        / "mdp"
        / "randomization.py"
    ).read_text()
    for needle in (
        "self.wheel_radius = 0.575",
        "self.vel_range = cfg.params[\"velocity_range\"]",
        "sampled_velocity_rads = (sampled_velocity_kmh / 3.6) / self.wheel_radius",
        "self.asset.write_joint_velocity_to_sim",
        "current_episode_lengths <= 10",
    ):
        if needle not in source:
            raise AssertionError(f"missing source evidence: {needle}")


def main() -> None:
    test_arm_and_vehicle_joint_conventions_match_obstacle_task()
    test_task_space_reference_constants_match_obstacle_task()
    test_vehicle_speed_conversion_matches_rrlab_randomizer()
    test_kevin_source_keeps_vehicle_command_separate_from_rom_action_contract()
    test_rrlab_obstacle_randomizer_source_uses_scripted_wheel_velocity()
    print("PASS: RRLAB reference convention tests passed.")


if __name__ == "__main__":
    main()
