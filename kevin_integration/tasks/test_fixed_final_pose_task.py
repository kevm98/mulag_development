"""Offline contracts for the fixed final task-space pose experiment.

These tests import production Torch helpers and inspect production source.
They do not require IsaacLab, Conda, or simulator startup.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "kevin_integration/tasks/mulag_adaptive_rl_env.py"
CFG_PATH = ROOT / "kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py"
REWARD_PATH = ROOT / "kevin_integration/rl/reward_terms.py"
OBS_PATH = ROOT / "kevin_integration/rl/observation_builder.py"
ACTION_PATH = ROOT / "kevin_integration/rl/action_adapter.py"
REFERENCE_PATH = ROOT / "kevin_integration/tasks/task_space_reference.py"
CONVENTIONS_PATH = ROOT / "kevin_integration/tasks/rrlab_reference_conventions.py"
CONTROLLER_PATH = ROOT / "kevin_integration/rl/adaptive_rl_controller.py"
MODELS_PATH = ROOT / "kevin_integration/controllers/learned_models.py"
ROM_PATH = ROOT / "kevin_integration/rom_cylinders/valve_conditioned_causal_rom.py"
ISAAC_DIRECT_ENV_PATH = (
    Path("/home/rrlab/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py")
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reward_terms = _load_module("fixed_pose_reward_terms", REWARD_PATH)
observation_builder = _load_module("fixed_pose_observation_builder", OBS_PATH)
action_adapter = _load_module("fixed_pose_action_adapter", ACTION_PATH)
task_reference = _load_module("fixed_pose_task_reference", REFERENCE_PATH)
conventions = _load_module("fixed_pose_conventions", CONVENTIONS_PATH)

ENV_SOURCE = ENV_PATH.read_text(encoding="utf-8")
CFG_SOURCE = CFG_PATH.read_text(encoding="utf-8")
ENV_TREE = ast.parse(ENV_SOURCE)
CFG_TREE = ast.parse(CFG_SOURCE)


def _class_node(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name} not found")


def _method_node(name: str) -> ast.FunctionDef:
    env_class = _class_node(ENV_TREE, "MulagAdaptiveRLEnv")
    for node in env_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Method {name} not found")


def _method_source(name: str) -> str:
    source = ast.get_source_segment(ENV_SOURCE, _method_node(name))
    if source is None:
        raise AssertionError(f"Source for {name} not found")
    return source


def _literal_assignment(class_name: str, field: str):
    class_def = _class_node(CFG_TREE, class_name)
    for node in class_def.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == field:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{class_name}.{field} not found")


def _sampler_callers() -> list[str]:
    callers: list[str] = []
    env_class = _class_node(ENV_TREE, "MulagAdaptiveRLEnv")
    for method in env_class.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_sample_task_space_command"
            ):
                callers.append(method.name)
    return callers


class TestFixedFinalPoseTask(unittest.TestCase):
    def test_target_contract_shape_order_and_frame(self) -> None:
        self.assertEqual(task_reference.TASK_SPACE_COMMAND_DIM, 7)
        self.assertEqual(
            task_reference.TASK_SPACE_POSE_ORDER,
            ("x", "y", "z", "qw", "qx", "qy", "qz"),
        )
        self.assertEqual(task_reference.TASK_SPACE_FRAME, "robot_base")
        target_source = _method_source("_final_pose_target")
        self.assertIn("final_pose_target = self._ee_pose_command", target_source)
        self.assertIn("(self.num_envs, 7)", target_source)
        self.assertIn("Final pose target must have shape", target_source)

    def test_reset_only_sampling_and_partial_reset(self) -> None:
        self.assertEqual(_sampler_callers(), ["_reset_idx"])
        reset_source = _method_source("_reset_idx")
        sampler_source = _method_source("_sample_task_space_command")
        self.assertIn("self._sample_task_space_command(env_ids)", reset_source)
        self.assertIn("final_pose_target[env_ids, 0]", sampler_source)
        self.assertIn("final_pose_target[env_ids, 3:7]", sampler_source)
        self.assertIn(
            "self._final_pose_target_at_episode_start[env_ids]",
            reset_source,
        )
        for method in (
            "_get_observations",
            "_get_rewards",
            "_get_task_space_rewards",
            "_pre_physics_step",
            "_apply_action",
            "_update_task_space_state",
        ):
            self.assertNotIn("_sample_task_space_command(", _method_source(method))

    def test_authoritative_target_drives_observation_and_errors(self) -> None:
        observation_source = _method_source("_get_observations")
        state_source = _method_source("_update_task_space_state")
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("self._final_pose_target()", observation_source)
        self.assertIn("target_position = final_pose_target[:, 0:3]", state_source)
        self.assertIn("position_error = target_position - current_position", state_source)
        self.assertIn("quaternion_geodesic_error(", state_source)
        self.assertIn("self._ee_position_error", reward_source)
        self.assertIn("self._ee_orientation_error", reward_source)
        self.assertNotIn("_raw_ee_pose_command", ENV_SOURCE)
        self.assertNotIn("_governed_ee_pose_command", ENV_SOURCE)

    def test_target_fixed_episode_debug_contract(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("if self.cfg.debug_reward_terms:", reward_source)
        self.assertIn(
            "self._final_pose_target_at_episode_start",
            reward_source,
        )
        self.assertIn('"final_pose/target_changed_fraction"', reward_source)

    def test_quaternion_geodesic_reference_cases(self) -> None:
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        negative_identity = -identity
        ninety_z = torch.tensor(
            [[math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]]
        )
        one_eighty_z = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        self.assertAlmostEqual(
            reward_terms.quaternion_geodesic_error(identity, identity).item(),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            reward_terms.quaternion_geodesic_error(identity, negative_identity).item(),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            reward_terms.quaternion_geodesic_error(identity, ninety_z).item(),
            math.pi / 2.0,
            places=5,
        )
        self.assertAlmostEqual(
            reward_terms.quaternion_geodesic_error(identity, one_eighty_z).item(),
            math.pi,
            places=5,
        )

    def test_quaternion_validation_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "norm <= 1e-8"):
            reward_terms.quaternion_geodesic_error(
                torch.zeros((1, 4)),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            )
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            reward_terms.normalize_quaternion_wxyz(
                torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]),
                name="test quaternion",
            )
        preflight_source = _method_source("run_task_space_bridge_preflight")
        self.assertEqual(preflight_source.count("validate=True"), 2)

    def test_orientation_reward_is_dense_nonzero_and_monotonic(self) -> None:
        errors = torch.tensor([0.0, 0.02, 0.08, 0.20])
        rewards = reward_terms.final_pose_orientation_reward(
            errors, 0.0872664626
        )
        self.assertTrue(bool(torch.isfinite(rewards).all()))
        self.assertTrue(bool(torch.all(rewards > 0.0)))
        self.assertTrue(bool(torch.all(rewards[:-1] > rewards[1:])))

    def test_reward_weights_and_single_inclusion(self) -> None:
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg", "rew_task_pos_tracking"
            ),
            1.0,
        )
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg", "rew_task_orient_tracking"
            ),
            0.25,
        )
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "final_pose_success_bonus_value",
            ),
            10.0,
        )
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("pos_tracking = torch.exp(-4.0 * pos_error_norm)", reward_source)
        reward_assignment = next(
            node
            for node in ast.walk(_method_node("_get_task_space_rewards"))
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "reward" for t in node.targets)
        )
        reward_names = [
            node.id for node in ast.walk(reward_assignment.value) if isinstance(node, ast.Name)
        ]
        self.assertEqual(reward_names.count("orient_tracking"), 1)
        self.assertEqual(reward_names.count("success_bonus"), 1)

    def test_success_requires_pose_and_settling(self) -> None:
        base = dict(
            position_tolerance_m=0.05,
            orientation_tolerance_rad=0.0872664626,
            linear_speed_tolerance_mps=0.05,
            angular_speed_tolerance_radps=0.10,
        )

        def within(pos: float, orient: float, linear: float, angular: float) -> bool:
            result = reward_terms.final_pose_within_tolerance(
                torch.tensor([pos]),
                torch.tensor([orient]),
                torch.tensor([linear]),
                torch.tensor([angular]),
                **base,
            )
            return bool(result.item())

        self.assertFalse(within(0.01, 0.20, 0.0, 0.0))
        self.assertFalse(within(0.20, 0.01, 0.0, 0.0))
        self.assertFalse(within(0.01, 0.01, 0.10, 0.0))
        self.assertFalse(within(0.01, 0.01, 0.0, 0.20))
        self.assertTrue(within(0.01, 0.01, 0.01, 0.01))

    def test_consecutive_hold_counter_and_reset(self) -> None:
        counter = torch.zeros(2, dtype=torch.long)
        valid = torch.tensor([True, False])
        counter = reward_terms.update_consecutive_success_counter(counter, valid)
        self.assertTrue(torch.equal(counter, torch.tensor([1, 0])))
        counter = reward_terms.update_consecutive_success_counter(
            counter, torch.tensor([False, True])
        )
        self.assertTrue(torch.equal(counter, torch.tensor([0, 1])))
        counter = torch.zeros(1, dtype=torch.long)
        for _ in range(50):
            counter = reward_terms.update_consecutive_success_counter(
                counter, torch.tensor([True])
            )
        self.assertEqual(counter.item(), 50)
        self.assertGreaterEqual(counter.item(), 50)

    def test_success_separate_from_unsafe_and_timeout(self) -> None:
        dones_source = _method_source("_get_dones")
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn(
            "terminated = unsafe_terminated | self._termination_success",
            dones_source,
        )
        self.assertIn("time_out = time_out & ~terminated", dones_source)
        unsafe_block = reward_source[
            reward_source.index("unsafe_termination = ("):
            reward_source.index("# ── Total reward")
        ]
        self.assertNotIn("_termination_success", unsafe_block)
        self.assertIn("self._within_final_pose = within_final_pose & ~unsafe_termination", ENV_SOURCE)

    def test_success_bonus_first_transition_and_partial_reset(self) -> None:
        success_source = _method_source("_update_final_pose_success_state")
        reset_source = _method_source("_reset_idx")
        self.assertIn(
            "final_pose_success & ~self._final_pose_success_previous",
            success_source,
        )
        for field in (
            "_final_pose_success_counter[env_ids] = 0",
            "_final_pose_success_previous[env_ids] = False",
            "_new_final_pose_success[env_ids] = False",
        ):
            self.assertIn(field, reset_source)

    def test_fixed_stage_one_configuration(self) -> None:
        base_class = "MulagAdaptiveRLJointReachEnvCfg"
        task_class = "MulagHydraulicROMTaskSpaceEnvCfg"
        self.assertFalse(_literal_assignment(base_class, "enable_automatic_curriculum"))
        self.assertFalse(_literal_assignment(task_class, "enable_automatic_curriculum"))
        self.assertEqual(
            _literal_assignment(task_class, "curriculum_action_authority")[0],
            0.25,
        )
        self.assertEqual(
            _literal_assignment(task_class, "curriculum_episode_length_s")[0],
            15.0,
        )
        curriculum_source = _method_source("_update_curriculum_state")
        self.assertIn("self._curriculum_stage_index = 1", curriculum_source)
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn('"curriculum/automatic_advancement_enabled"', reward_source)

    def test_exact_final_pose_tolerances(self) -> None:
        class_name = "MulagAdaptiveRLJointReachEnvCfg"
        expected = {
            "final_pose_position_tolerance_m": 0.05,
            "final_pose_orientation_tolerance_rad": 0.0872664626,
            "final_pose_linear_speed_tolerance_mps": 0.05,
            "final_pose_angular_speed_tolerance_radps": 0.10,
            "final_pose_success_hold_steps": 50,
        }
        for field, value in expected.items():
            self.assertEqual(_literal_assignment(class_name, field), value)

    def test_observation_and_action_dimensions_remain_fixed(self) -> None:
        self.assertEqual(task_reference.TASK_SPACE_OBSERVATION_DIM, 73)
        tensors = {
            "ee_pose_command": torch.zeros(2, 7),
            "current_ee_pose": torch.zeros(2, 7),
            "ee_position_error": torch.zeros(2, 3),
            "ee_orientation_error": torch.zeros(2, 1),
            "mechanical_q": torch.zeros(2, 5),
            "mechanical_qdot": torch.zeros(2, 5),
            "cylinder_q": torch.zeros(2, 4),
            "cylinder_qdot": torch.zeros(2, 4),
            "prev_action": torch.zeros(2, 4),
            "prev_valve_cmd": torch.zeros(2, 4),
            "z_arm_hat": torch.zeros(2, 24),
            "fd_tau_hat": torch.zeros(2, 4),
            "vehicle_speed_kmh": torch.zeros(2, 1),
        }
        observation = observation_builder.build_task_space_rl_obs(**tensors)
        self.assertEqual(observation.shape, (2, 73))
        self.assertEqual(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "action_space"),
            4,
        )
        qddot = action_adapter.expand_action_to_id(torch.ones(2, 4))
        self.assertEqual(qddot.shape, (2, 5))
        self.assertTrue(bool(torch.all(qddot[:, 4] == 0.0)))

    def test_hydraulic_dimensions_and_follower_ownership_unchanged(self) -> None:
        controller_source = CONTROLLER_PATH.read_text(encoding="utf-8")
        self.assertIn('"id_output": "ValveCmd[4]"', controller_source)
        self.assertIn("num_cylinders=4", ENV_SOURCE)
        self.assertIn("output_dim=4", ENV_SOURCE)
        self.assertIn(
            "set_joint_position_target(self._rom_cylinder_pos_target, "
            "joint_ids=self._command_joint_ids)",
            ENV_SOURCE,
        )
        self.assertEqual(len(conventions.OBSTACLE_ARM_JOINT_NAMES), 5)
        self.assertEqual(len(conventions.OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES), 4)
        self.assertEqual(
            conventions.OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES[-1],
            "Messerkopf_Schwenk_joint",
        )
        self.assertNotIn(
            "Messerkopf_joint",
            conventions.OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES,
        )

    def test_models_remain_frozen(self) -> None:
        controller_source = CONTROLLER_PATH.read_text(encoding="utf-8")
        model_source = MODELS_PATH.read_text(encoding="utf-8")
        rom_source = ROM_PATH.read_text(encoding="utf-8")
        self.assertIn("wrapper.model.requires_grad_(False)", controller_source)
        self.assertIn("wrapper.model.eval()", controller_source)
        self.assertGreaterEqual(model_source.count("self.model.eval()"), 4)
        self.assertGreaterEqual(model_source.count("with torch.no_grad():"), 4)
        self.assertIn("self.model.eval()", rom_source)
        self.assertIn("@torch.no_grad()", rom_source)

    def test_terminal_snapshots_and_metrics_are_present(self) -> None:
        reset_source = _method_source("_reset_idx")
        super_index = reset_source.index("super()._reset_idx(env_ids)")
        for field in (
            "final_pose_success_snapshot",
            "final_position_error_snapshot",
            "final_orientation_error_snapshot",
            "final_linear_speed_snapshot",
            "final_angular_speed_snapshot",
            "final_success_counter_snapshot",
            "episode_length_snapshot",
            "termination_reason_snapshot",
        ):
            self.assertLess(reset_source.index(field), super_index)
        for key in (
            "Episode/final_pose_success",
            "Episode/final_position_error",
            "Episode/final_orientation_error_rad",
            "Episode/final_orientation_error_deg",
            "Episode/final_linear_speed",
            "Episode/final_angular_speed",
            "Episode/final_success_hold_steps",
            "Episode/time_to_success_steps",
        ):
            self.assertIn(key, reset_source)

    def test_direct_rl_env_lifecycle_order(self) -> None:
        if not ISAAC_DIRECT_ENV_PATH.is_file():
            self.skipTest("Installed IsaacLab DirectRLEnv source is unavailable")
        source = ISAAC_DIRECT_ENV_PATH.read_text(encoding="utf-8")
        dones = source.index("self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()")
        rewards = source.index("self.reward_buf = self._get_rewards()")
        reset = source.index("self._reset_idx(reset_env_ids)")
        self.assertLess(dones, rewards)
        self.assertLess(rewards, reset)

    def test_hydra_and_launcher_files_have_no_worktree_diff(self) -> None:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--",
                "kevin_integration/tasks/__init__.py",
                "kevin_integration/scripts/train_mulag_adaptive_rl.py",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
