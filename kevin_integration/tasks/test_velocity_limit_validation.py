"""Static velocity-limit contracts for the Kevin hydraulic task.

These tests avoid IsaacLab startup and inspect the production source for the
validated-velocity audit contract.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "kevin_integration/tasks/mulag_adaptive_rl_env.py"
CFG_PATH = ROOT / "kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py"
SCRIPT_PATH = ROOT / "kevin_integration/scripts/analyze_simscape_velocity_limits.py"
CONVENTIONS_PATH = ROOT / "kevin_integration/tasks/rrlab_reference_conventions.py"

ENV_SOURCE = ENV_PATH.read_text(encoding="utf-8")
CFG_SOURCE = CFG_PATH.read_text(encoding="utf-8")
SCRIPT_SOURCE = SCRIPT_PATH.read_text(encoding="utf-8")
ENV_TREE = ast.parse(ENV_SOURCE)
CFG_TREE = ast.parse(CFG_SOURCE)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


conventions = _load_module("velocity_limit_conventions", CONVENTIONS_PATH)


def _class_node(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name} not found")


def _method_source(name: str) -> str:
    env_class = _class_node(ENV_TREE, "MulagAdaptiveRLEnv")
    for node in env_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            source = ast.get_source_segment(ENV_SOURCE, node)
            if source is None:
                raise AssertionError(f"Source for {name} not found")
            return source
    raise AssertionError(f"Method {name} not found")


def _literal_assignment(class_name: str, field: str):
    class_def = _class_node(CFG_TREE, class_name)
    for node in class_def.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == field:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{class_name}.{field} not found")


class TestVelocityLimitValidation(unittest.TestCase):
    def test_joint_and_hydraulic_channel_order(self) -> None:
        self.assertEqual(
            conventions.OBSTACLE_ARM_JOINT_NAMES,
            (
                "Drehzapfen_joint",
                "Ausleger_I_joint",
                "Ausleger_II_joint",
                "Messerkopf_Schwenk_joint",
                "Messerkopf_joint",
            ),
        )
        self.assertEqual(
            conventions.OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES,
            conventions.OBSTACLE_ARM_JOINT_NAMES[:4],
        )

    def test_old_threshold_is_diagnostic_only(self) -> None:
        self.assertEqual(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "unvalidated_velocity_diagnostic_threshold"),
            6.0,
        )
        dones_source = _method_source("_get_dones")
        self.assertNotIn("max_abs_joint_velocity", CFG_SOURCE)
        self.assertNotIn("self.cfg.max_abs_joint_velocity", dones_source)
        self.assertIn("enable_validated_hydraulic_velocity_termination", CFG_SOURCE)

    def test_per_channel_limits_enabled_with_conservative_rom_evidence(self) -> None:
        self.assertTrue(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "enable_validated_hydraulic_velocity_termination")
        )
        self.assertEqual(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "hydraulic_hard_velocity_limits_mps"),
            [0.5, 0.5, 0.5, 0.5],
        )
        self.assertEqual(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "hydraulic_velocity_limits_source"),
            "conservative_rom_operating_range_0p5_mps",
        )

    def test_mechanical_and_cylinder_contracts_are_explicit(self) -> None:
        self.assertIn("mechanical_joint_names = list(OBSTACLE_ARM_JOINT_NAMES)", CFG_SOURCE)
        self.assertIn("cylinder_joint_names = list(OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES)", CFG_SOURCE)
        self.assertEqual(len(conventions.OBSTACLE_ARM_JOINT_NAMES), 5)
        self.assertEqual(len(conventions.OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES), 4)
        self.assertEqual(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "mechanical_joint_velocity_units"),
            ["rad/s", "rad/s", "rad/s", "rad/s", "rad/s"],
        )
        self.assertEqual(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "cylinder_joint_velocity_units"),
            ["m/s", "m/s", "m/s", "m/s"],
        )

    def test_state_readers_are_separate(self) -> None:
        mechanical_source = _method_source("_read_mechanical_state")
        cylinder_source = _method_source("_read_cylinder_state")
        self.assertIn("self._mechanical_joint_ids", mechanical_source)
        self.assertIn("mechanical_qdot", mechanical_source)
        self.assertIn("self._cylinder_joint_ids", cylinder_source)
        self.assertIn("cylinder_extension_velocity", cylinder_source)

    def test_startup_assertions_document_units_shapes_and_overlap(self) -> None:
        source = _method_source("_assert_velocity_state_contract")
        self.assertIn("Mechanical learned-model state must resolve exactly five joints", source)
        self.assertIn("Cylinder ROM/writer state must resolve exactly four joints", source)
        self.assertIn("Mechanical qdot units must be rad/s", source)
        self.assertIn("Cylinder extension velocity units must be m/s", source)
        self.assertIn("_mechanical_cylinder_joint_id_overlap", source)

    def test_qdot_fd_target_velocity_and_rom_applied_rate(self) -> None:
        source = _method_source("_update_hydraulic_velocity_audit")
        self.assertIn("self._hydraulic_qdot_fd = (q_hyd - self._prev_hydraulic_q) / control_dt", source)
        self.assertIn("self._hydraulic_target_velocity = (", source)
        self.assertIn("self._rom_cylinder_pos_target - self._rom_cylinder_pos_target_prev", source)
        self.assertIn("self._hydraulic_rom_applied_rate = self._final_rom_delta_isaac_10ms / control_dt", source)

    def test_cylinder_velocity_audit_return_is_nonfinite_only(self) -> None:
        source = _method_source("_update_hydraulic_velocity_audit")
        self.assertIn("return ~finite_all", source)
        self.assertNotIn("return torch.any(confirmed", source)

    def test_persistence_and_corroboration_required_for_validated_hard_termination(self) -> None:
        source = _method_source("_update_hydraulic_velocity_audit")
        self.assertIn("hydraulic_velocity_persistence_steps", source)
        self.assertIn("corroborated", source)
        self.assertIn("self._hydraulic_velocity_persistence >= required_steps", source)
        self.assertIn("bool(self._validated_hydraulic_velocity_limits_enabled())", source)

    def test_fifth_follower_joint_not_using_hydraulic_limits(self) -> None:
        dones_source = _method_source("_get_dones")
        self.assertIn("mechanical_hard_exceedance[:, j]", dones_source)
        self.assertNotIn("self._hydraulic_velocity_confirmed_hard[:, j]", dones_source)

    def test_one_step_spike_is_logged_not_confirmed_by_default(self) -> None:
        source = _method_source("_update_hydraulic_velocity_audit")
        self.assertIn("self._hydraulic_velocity_spike_ignored", source)
        self.assertIn("corroborated = torch.zeros_like(raw_exceedance)", source)

    def test_velocity_audit_telemetry_keys(self) -> None:
        dones_source = _method_source("_get_dones")
        for key in (
            '"termination/velocity_raw_exceedance"',
            '"termination/velocity_confirmed_hard"',
            '"termination/velocity_spike_ignored"',
            '"termination/velocity_persistence_max"',
            '"velocity_audit/physx_spike_count"',
            '"velocity_audit/target_jump_count"',
            '"velocity_audit/genuine_high_motion_count"',
            '"velocity_audit/unit_mapping_mismatch_count"',
            '"velocity_audit/nonfinite_count"',
            '"mechanical_velocity/abs_mean_per_joint"',
            '"mechanical_velocity/abs_max_per_joint"',
            '"mechanical_velocity/soft_exceedance_fraction_per_joint"',
            '"mechanical_velocity/hard_exceedance_fraction_per_joint"',
            '"cylinder_velocity/abs_mean_per_channel"',
            '"cylinder_velocity/abs_max_per_channel"',
            '"cylinder_velocity/old_6_threshold_crossing_fraction"',
            '"cylinder_velocity/diagnostic_only"',
        ):
            self.assertIn(key, dones_source)

    def test_model_feature_routing_uses_mechanical_state_and_cylinder_rom_state(self) -> None:
        pre_source = _method_source("_pre_physics_step")
        rom_source = _method_source("_pre_physics_step_final_causal_rom")
        feature_source = _method_source("_final_causal_rom_feature_row")
        self.assertIn("mechanical_q, mechanical_qdot = self._read_mechanical_state()", pre_source)
        self.assertIn("self._maybe_initialize_learned_control(mechanical_q, mechanical_qdot, dP)", pre_source)
        self.assertIn("q=mechanical_q", pre_source)
        self.assertIn("qdot=mechanical_qdot", pre_source)
        self.assertIn("q_cyl = self.robot.data.joint_pos[:, self._command_joint_ids]", rom_source)
        self.assertIn("qdot_cyl = self.robot.data.joint_vel[:, self._command_joint_ids]", rom_source)
        self.assertIn("JointPos_1..5", feature_source)
        self.assertIn("ExtensionVelocity_1..4", feature_source)

    def test_export_method_is_bounded_and_cpu_readable(self) -> None:
        source = _method_source("export_velocity_audit_events")
        self.assertIn("max_events = min(int(env_ids.numel()), 32)", source)
        self.assertIn('"qdot_physx"', source)
        self.assertIn('"target_velocity"', source)
        self.assertIn('"rom_extension_velocity"', source)
        self.assertIn('"rom_applied_rate"', source)

    def test_simscape_script_declares_velocity_units_and_required_fields(self) -> None:
        for field in ("Time", "Extension", "ExtensionVelocity", "JointVel"):
            self.assertIn(field, SCRIPT_SOURCE)
        self.assertIn('"ExtensionVelocity": "m/s"', SCRIPT_SOURCE)
        self.assertIn("rad/s for revolute coordinates and m/s for prismatic", SCRIPT_SOURCE)
        self.assertIn("pytorch_hydraulic_rom_dataset_causal.mat", SCRIPT_SOURCE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
