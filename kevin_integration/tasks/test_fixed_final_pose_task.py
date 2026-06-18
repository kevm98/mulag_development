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
from types import SimpleNamespace

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
TRAIN_PATH = ROOT / "kevin_integration/scripts/train_mulag_adaptive_rl.py"
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
TRAIN_SOURCE = TRAIN_PATH.read_text(encoding="utf-8")
ENV_TREE = ast.parse(ENV_SOURCE)
CFG_TREE = ast.parse(CFG_SOURCE)
TRAIN_TREE = ast.parse(TRAIN_SOURCE)


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


def _train_curriculum_namespace(args_cli: SimpleNamespace) -> dict[str, object]:
    helper_names = {
        "_is_task_space_training_cfg",
        "_has_explicit_q_ref_scale_override",
        "_apply_goal_difficulty_curriculum_default",
    }
    helper_source = "\n\n".join(
        ast.get_source_segment(TRAIN_SOURCE, node) or ""
        for node in ast.walk(TRAIN_TREE)
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    )
    namespace: dict[str, object] = {"args_cli": args_cli}
    exec("from __future__ import annotations\n" + helper_source, namespace)
    return namespace


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
        self.assertIn("final_pose_target[env_ids]", sampler_source)
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

    def test_reset_sampler_generates_reachable_pose_from_controlled_configuration(self) -> None:
        sampler_source = _method_source("_sample_task_space_command")
        self.assertIn("sampled_joint_pos = self._q_ref_target[env_ids]", sampler_source)
        self.assertIn("self._apply_joint5_coupling_to_reference(", sampler_source)
        self.assertIn("joint_pos_for_fk[:, self._state_joint_ids] = sampled_joint_pos", sampler_source)
        self.assertIn("self.robot.write_joint_state_to_sim(joint_pos_for_fk", sampler_source)
        self.assertIn("self.robot.update(0.0)", sampler_source)
        self.assertIn("self._update_task_space_state()", sampler_source)
        self.assertIn(
            "final_pose_target[env_ids] = self._current_ee_pose[env_ids]",
            sampler_source,
        )
        self.assertNotIn("curriculum_command_x_range", sampler_source)
        self.assertNotIn("uniform_(-0.05, 0.05)", sampler_source)

    def test_reachable_sampler_uses_stage_one_safe_joint_ranges(self) -> None:
        sampler_source = _method_source("_sample_joint_targets")
        reset_source = _method_source("_reset_idx")
        self.assertIn("self._sample_joint_targets(env_ids, q_ref_center)", reset_source)
        self.assertIn("self.cfg.q_ref_target_amplitude", sampler_source)
        self.assertIn("self._current_q_ref_target_scale()", sampler_source)
        self.assertIn("safe_lower", sampler_source)
        self.assertIn("safe_upper", sampler_source)
        self.assertIn("self._apply_joint5_coupling_to_reference(q_ref_target", sampler_source)

    def test_goal_curriculum_starts_at_easiest_reachable_stage(self) -> None:
        self.assertFalse(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "enable_goal_difficulty_curriculum",
            )
        )
        self.assertEqual(
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "goal_curriculum_stage_scales")[0],
            0.25,
        )
        self.assertIn("self._curriculum_stage_index = 1", ENV_SOURCE)
        scale_source = _method_source("_current_q_ref_target_scale")
        self.assertIn('getattr(self.cfg, "enable_goal_difficulty_curriculum", False)', scale_source)
        self.assertIn("scales[stage - 1]", scale_source)

    def test_fresh_training_enables_goal_curriculum_without_checkpoint(self) -> None:
        class EnvCfg:
            control_mode = "rl_task_space_tracking"
            reference_mode = "rl_task_space_tracking"
            enable_goal_difficulty_curriculum = False

        args = SimpleNamespace(
            checkpoint=None,
            enable_goal_difficulty_curriculum=False,
            disable_goal_difficulty_curriculum=False,
            q_ref_target_scale=None,
            q_ref_target_scale_initial=None,
            q_ref_target_scale_final=None,
            q_ref_curriculum_steps=None,
        )
        namespace = _train_curriculum_namespace(args)
        env_cfg = EnvCfg()
        namespace["_apply_goal_difficulty_curriculum_default"](env_cfg)
        self.assertTrue(env_cfg.enable_goal_difficulty_curriculum)

    def test_checkpoint_training_does_not_auto_enable_goal_curriculum(self) -> None:
        class EnvCfg:
            control_mode = "rl_task_space_tracking"
            reference_mode = "rl_task_space_tracking"
            enable_goal_difficulty_curriculum = False

        args = SimpleNamespace(
            checkpoint="/tmp/model.pt",
            enable_goal_difficulty_curriculum=False,
            disable_goal_difficulty_curriculum=False,
            q_ref_target_scale=None,
            q_ref_target_scale_initial=None,
            q_ref_target_scale_final=None,
            q_ref_curriculum_steps=None,
        )
        namespace = _train_curriculum_namespace(args)
        env_cfg = EnvCfg()
        namespace["_apply_goal_difficulty_curriculum_default"](env_cfg)
        self.assertFalse(env_cfg.enable_goal_difficulty_curriculum)

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
        target_changed_source = _method_source("_target_changed_this_episode")
        self.assertIn("if self.cfg.debug_reward_terms:", reward_source)
        self.assertIn(
            "self._final_pose_target_at_episode_start",
            target_changed_source,
        )
        self.assertIn("target_changed = self._target_changed_this_episode()", reward_source)
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
                "MulagAdaptiveRLJointReachEnvCfg", "position_tracking_reward_weight"
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
        self.assertIn("W_pos = float(self.cfg.position_tracking_reward_weight)", reward_source)
        self.assertIn("pos_tracking = torch.exp(-4.0 * pos_error_norm)", reward_source)
        self.assertIn("pos_tracking_weighted = W_pos * pos_tracking", reward_source)
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
        self.assertEqual(reward_names.count("linear_velocity_penalty"), 1)
        self.assertEqual(reward_names.count("angular_velocity_penalty"), 1)
        self.assertEqual(reward_names.count("settling_reward"), 1)
        self.assertEqual(reward_names.count("hold_reward"), 1)

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
        self.assertIn("self._termination_velocity", unsafe_block)
        self.assertIn("self._termination_lower_limit", unsafe_block)
        self.assertIn("self._termination_upper_limit", unsafe_block)
        self.assertIn("self._termination_joint_nonfinite", unsafe_block)
        self.assertIn("self._termination_control_nonfinite", unsafe_block)
        self.assertIn("& ~self._termination_success", unsafe_block)
        self.assertIn("& ~self.reset_time_outs", unsafe_block)
        self.assertIn("-10.0 * W_pos", unsafe_block)
        self.assertNotIn("unsafe_terminal_penalty", unsafe_block)
        self.assertIn("self._within_final_pose = strict_all_except_hold.detach().clone()", ENV_SOURCE)

    def test_near_target_velocity_shaping_terms_and_telemetry(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        for snippet in (
            "position_gate = torch.exp(",
            "pos_error_norm / 0.20",
            "orientation_gate = torch.exp(",
            "orient_error / 0.1745329252",
            "pose_gate = position_gate * orientation_gate",
            "ee_linear_speed = self._ee_linear_speed.reshape(-1)",
            "ee_angular_speed = self._ee_angular_speed.reshape(-1)",
            "float(self.cfg.rew_final_pose_linear_velocity)",
            "ee_linear_speed / float(self.cfg.final_pose_linear_shaping_scale_mps)",
            "float(self.cfg.rew_final_pose_angular_velocity)",
            "ee_angular_speed / float(self.cfg.final_pose_angular_shaping_scale_radps)",
            "linear_settling_score = torch.exp(",
            "ee_linear_speed / float(self.cfg.final_pose_linear_settling_scale_mps)",
            "angular_settling_score = torch.exp(",
            "ee_angular_speed / float(self.cfg.final_pose_angular_settling_scale_radps)",
            "float(self.cfg.rew_final_pose_settling)",
            "hold_reward = hold_progress_reward(",
            "reward/linear_velocity_penalty_mean",
            "reward/angular_velocity_penalty_mean",
            "reward/settling_reward_mean",
            "reward/hold_progress_reward_mean",
            "episode/linear_velocity_penalty_sum",
            "episode/angular_velocity_penalty_sum",
            "episode/settling_reward_sum",
            "episode/hold_progress_reward_sum",
            "debug/velocity_mask_count",
            "debug/unsafe_mask_count",
            "debug/unsafe_penalty_nonzero_count",
            "debug/unsafe_penalty_min",
        ):
            self.assertIn(snippet, reward_source)
        for absent in (
            "pos_error_norm / 0.05",
            "orient_error / 0.0872664626",
        ):
            self.assertNotIn(absent, reward_source)

    def test_action_smoothing_uses_normalized_task_space_formulas(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("action_delta = (", reward_source)
        self.assertIn("self._current_action", reward_source)
        self.assertIn("self._previous_action", reward_source)
        self.assertIn("-0.01", reward_source)
        self.assertIn("torch.mean(\n                torch.square(action_delta),", reward_source)
        self.assertIn("-0.0025", reward_source)
        self.assertIn("torch.square(\n                    self._current_action", reward_source)
        self.assertNotIn("float(self.cfg.rew_action_rate) * torch.mean", reward_source)
        self.assertNotIn("float(self.cfg.rew_action_magnitude) * torch.mean", reward_source)

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
        success_source = _method_source("_update_final_pose_success_state")
        for field in expected:
            self.assertIn(f"self.cfg.{field}", success_source)
        self.assertNotIn("_curriculum_stage", success_source)

    def test_required_hold_telemetry_keys_are_logged(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        rolling_source = _method_source("_rolling_1000_log")
        for key in (
            '"hold_progress_reward_mean"',
            '"settling_reward_mean"',
            '"final_pose/success_hold_counter_mean"',
            '"final_pose/settling_hold_max_total"',
            '"final_pose/near_pose_low_speed_fraction"',
            '"final_pose/strict_position_fraction"',
            '"final_pose/strict_orientation_fraction"',
            '"final_pose/strict_linear_speed_fraction"',
            '"final_pose/strict_angular_speed_fraction"',
            '"final_pose/strict_all_except_hold_fraction"',
            '"final_pose/hold_break_reason_position"',
            '"final_pose/hold_break_reason_orientation"',
            '"final_pose/hold_break_reason_linear_speed"',
            '"final_pose/hold_break_reason_angular_speed"',
            '"final_pose/hold_break_reason_target_changed"',
            '"final_pose/hold_break_reason_timeout"',
            '"final_pose/hold_break_reason_success_reached"',
            '"final_pose/hold_break_reason_unknown"',
            '"final_pose/max_hold_position_error_at_break"',
            '"final_pose/max_hold_orientation_error_at_break"',
            '"final_pose/max_hold_linear_speed_at_break"',
            '"final_pose/max_hold_angular_speed_at_break"',
            '"final_pose/max_hold_steps_before_break"',
            '"final_pose/steps_remaining_after_max_hold"',
            '"final_pose/first_step_entered_strict_settled"',
            '"final_pose/last_step_entered_strict_settled"',
        ):
            self.assertIn(key, reward_source)
        self.assertIn('"rolling_1000/max_hold_steps"', rolling_source)
        self.assertIn("_rolling_1000_max(self._rolling_1000_max_hold_steps)", rolling_source)

    def test_hold_counter_uses_only_four_strict_conditions(self) -> None:
        success_source = _method_source("_update_final_pose_success_state")
        counter_block = success_source[
            success_source.index("strict_position =") :
            success_source.index("final_pose_success = (")
        ]
        for snippet in (
            "strict_position",
            "strict_orientation",
            "strict_linear_speed",
            "strict_angular_speed",
            "strict_all_except_hold = (",
            "self._within_final_pose = strict_all_except_hold.detach().clone()",
            "update_consecutive_success_counter(",
        ):
            self.assertIn(snippet, counter_block)
        update_call = counter_block[
            counter_block.index("self._final_pose_success_counter = update_consecutive_success_counter(") :
            counter_block.index("new_env_max =")
        ]
        self.assertIn("self._within_final_pose", update_call)
        self.assertNotIn("unsafe_termination", update_call)

    def test_hold_break_records_failed_strict_condition_reasons(self) -> None:
        success_source = _method_source("_update_final_pose_success_state")
        self.assertIn("hold_broke = (previous_counter > 0) & ~strict_all_except_hold", success_source)
        for snippet in (
            "reason_position=hold_broke & ~strict_position",
            "reason_orientation=hold_broke & ~strict_orientation",
            "reason_linear_speed=hold_broke & ~strict_linear_speed",
            "reason_angular_speed=hold_broke & ~strict_angular_speed",
            "reason_target_changed=hold_broke & target_changed",
        ):
            self.assertIn(snippet, success_source)

    def test_success_reached_suppresses_unknown_hold_break_reason(self) -> None:
        ongoing_source = _method_source("_update_global_hold_diagnostic_without_break")
        break_source = _method_source("_record_hold_break_diagnostic")
        for source in (ongoing_source, break_source):
            self.assertIn(
                "success_reached = best_steps >= int(self.cfg.final_pose_success_hold_steps)",
                source,
            )
            self.assertIn(
                "self._final_pose_hold_break_reason_success_reached.copy_(",
                source,
            )
        self.assertIn(
            "self._final_pose_hold_break_reason_unknown.copy_((~success_reached).detach().float())",
            ongoing_source,
        )
        self.assertIn("| success_reached", break_source)
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn('"final_pose/hold_break_reason_success_reached"', reward_source)

    def test_success_is_counted_before_timeout_and_timeout_cannot_erase_it(self) -> None:
        dones_source = _method_source("_get_dones")
        self.assertLess(
            dones_source.index("self._update_final_pose_success_state(unsafe_terminated)"),
            dones_source.index("time_out = self.episode_length_buf"),
        )
        self.assertLess(
            dones_source.index("terminated = unsafe_terminated | self._termination_success"),
            dones_source.index("time_out = time_out & ~terminated"),
        )
        self.assertIn("time_out = time_out & ~terminated", dones_source)
        self.assertIn("self._termination_success_count += self._new_final_pose_success.float()", dones_source)

    def test_timeout_break_is_recorded_after_timeout_masking(self) -> None:
        dones_source = _method_source("_get_dones")
        timeout_mask = dones_source.index("time_out = time_out & ~terminated")
        timeout_break = dones_source.index("self._record_hold_timeout_break(time_out)")
        self.assertLess(timeout_mask, timeout_break)
        timeout_source = _method_source("_record_hold_timeout_break")
        self.assertIn("timeout_break = time_out & (current_hold_steps > 0) & ~self._termination_success", timeout_source)
        self.assertIn("reason_timeout=timeout_break", timeout_source)

    def test_target_is_fixed_and_episode_has_room_after_first_strict_entry(self) -> None:
        self.assertEqual(_sampler_callers(), ["_reset_idx"])
        self.assertEqual(
            _literal_assignment("MulagHydraulicROMTaskSpaceEnvCfg", "curriculum_episode_length_s")[0],
            15.0,
        )
        self.assertGreater(
            int(15.0 / 0.01),
            _literal_assignment("MulagAdaptiveRLJointReachEnvCfg", "final_pose_success_hold_steps"),
        )
        success_source = _method_source("_update_final_pose_success_state")
        self.assertIn("self._final_pose_first_strict_settled_step", success_source)
        self.assertIn("self._final_pose_last_strict_settled_step", success_source)
        self.assertIn("self._final_pose_steps_remaining_after_max_hold", success_source)

    def test_observation_and_action_dimensions_remain_fixed(self) -> None:
        self.assertEqual(task_reference.TASK_SPACE_OBSERVATION_DIM, 70)
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
        self.assertEqual(observation.shape, (2, 70))
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
            "final_max_hold_snapshot",
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
            "Episode/max_hold_steps",
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

    def test_hydra_source_is_not_modified_and_task_registration_is_local(self) -> None:
        hydra_path = Path("/home/rrlab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py")
        isaac_root = Path("/home/rrlab/IsaacLab")
        if hydra_path.is_file() and (isaac_root / ".git").is_dir():
            result = subprocess.run(
                ["git", "diff", "--", str(hydra_path.relative_to(isaac_root))],
                cwd=isaac_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout, "")
        task_init = (ROOT / "kevin_integration/tasks/__init__.py").read_text(encoding="utf-8")
        self.assertIn('id="Kevin-Mulag-HydraulicROM-TaskSpace-v0"', task_init)


class TestUnsafePenalty(unittest.TestCase):
    """Unsafe-penalty wiring: shape assertion, value, debug keys, exclusion."""

    def test_unsafe_penalty_shape_assertion_present(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("if unsafe_termination.shape != (self.num_envs,):", reward_source)
        self.assertIn("raise RuntimeError(", reward_source)
        self.assertIn("unsafe_termination mask must have shape", reward_source)

    def test_unsafe_penalty_value_is_minus_10_times_W_pos(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("-10.0 * W_pos", reward_source)

    def test_unsafe_penalty_excludes_success_and_timeout(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("& ~self._termination_success", reward_source)
        self.assertIn("& ~self.reset_time_outs", reward_source)

    def test_unsafe_penalty_uses_all_five_termination_masks(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        unsafe_block = reward_source[
            reward_source.index("unsafe_termination = (") :
            reward_source.index("# ── Debug telemetry for unsafe penalty")
        ]
        for mask in (
            "self._termination_velocity",
            "self._termination_lower_limit",
            "self._termination_upper_limit",
            "self._termination_joint_nonfinite",
            "self._termination_control_nonfinite",
        ):
            self.assertIn(mask, unsafe_block)

    def test_unsafe_penalty_debug_telemetry_scalars_computed(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        for scalar in (
            "velocity_mask_count = self._termination_velocity.sum()",
            "unsafe_mask_count = unsafe_termination.sum()",
            "unsafe_penalty_nonzero_count = (unsafe_penalty != 0.0).sum()",
            "unsafe_penalty_min = unsafe_penalty.min()",
        ):
            self.assertIn(scalar, reward_source)

    def test_unsafe_penalty_debug_keys_in_log_dict(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        for key in (
            '"debug/velocity_mask_count"',
            '"debug/unsafe_mask_count"',
            '"debug/unsafe_penalty_nonzero_count"',
            '"debug/unsafe_penalty_min"',
        ):
            self.assertIn(key, reward_source)

    def test_unsafe_penalty_accumulator_in_episode_sums(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("self._episode_unsafe_penalty_sum += unsafe_penalty.detach()", reward_source)
        self.assertIn('"episode/unsafe_penalty_sum"', reward_source)

    def test_unsafe_penalty_does_not_exceed_minus_10_bound(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertNotIn("-100.0", reward_source)
        self.assertNotIn("-20.0", reward_source)

    def test_unsafe_penalty_applied_via_torch_where(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("torch.where(", reward_source)
        self.assertIn("unsafe_termination", reward_source)

    def test_unsafe_penalty_masks_detached_and_cloned_in_dones(self) -> None:
        dones_source = _method_source("_get_dones")
        self.assertIn("self._termination_velocity = velocity_violation.detach().clone()", dones_source)

    def test_unsafe_mean_logged_as_reward_unsafe_penalty_mean(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn('"reward/unsafe_penalty_mean"', reward_source)


class TestSettlingReward(unittest.TestCase):
    """Settling reward uses decoupled velocity scores + pose_gate."""

    def test_settling_uses_linear_and_angular_scores(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("linear_settling_score = torch.exp(", reward_source)
        self.assertIn("angular_settling_score = torch.exp(", reward_source)

    def test_settling_linear_scale_uses_configured_final_pose_scale(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("ee_linear_speed / float(self.cfg.final_pose_linear_settling_scale_mps)", reward_source)

    def test_settling_angular_scale_uses_configured_final_pose_scale(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("ee_angular_speed / float(self.cfg.final_pose_angular_settling_scale_radps)", reward_source)

    def test_settling_multiplied_by_pose_gate(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        settling_block = reward_source[
            reward_source.index("settling_reward = (") :
            reward_source.index("── Penalties ──")
        ]
        self.assertIn("pose_gate", settling_block)

    def test_settling_uses_configured_weight(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        settling_block = reward_source[
            reward_source.index("settling_reward = (") :
            reward_source.index("── Penalties ──")
        ]
        self.assertIn("float(self.cfg.rew_final_pose_settling)", settling_block)

    def test_settling_weighted_by_W_pos(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        settling_block = reward_source[
            reward_source.index("settling_reward = (") :
            reward_source.index("── Penalties ──")
        ]
        self.assertIn("* W_pos", settling_block)

    def test_settling_reward_uses_broad_gate_and_velocity_scales_not_tolerances(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertNotIn("pos_error_norm / 0.05", reward_source)
        self.assertNotIn("orient_error / 0.0872664626", reward_source)

    def test_settling_reward_mean_in_log(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn('"reward/settling_reward_mean"', reward_source)

    def test_settling_episode_sum_in_log(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn('"episode/settling_reward_sum"', reward_source)

    def test_settling_accumulator_present(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("self._episode_settling_reward_sum += settling_reward.detach()", reward_source)

    def test_settling_reward_included_in_total_reward(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        total_block = reward_source[
            reward_source.index("# ── Total reward ──") :
            reward_source.index("# ── Update progress buffer")
        ]
        self.assertIn("settling_reward", total_block)

    def test_settling_gate_scales_are_0_point_20m_and_10_deg(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("pos_error_norm / 0.20", reward_source)
        self.assertIn("orient_error / 0.1745329252", reward_source)


class TestPreservation(unittest.TestCase):
    """Existing reward terms and pipeline are preserved."""

    def test_position_tracking_formula_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("pos_tracking = torch.exp(-4.0 * pos_error_norm)", reward_source)
        self.assertIn("pos_tracking_weighted = W_pos * pos_tracking", reward_source)

    def test_orientation_tracking_formula_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("orient_tracking_unweighted = final_pose_orientation_reward(", reward_source)
        self.assertIn("orient_error,", reward_source)
        self.assertIn("float(self.cfg.final_pose_orientation_tolerance_rad)", reward_source)

    def test_success_bonus_formula_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("success_bonus = torch.where(", reward_source)
        self.assertIn("self._new_final_pose_success", reward_source)
        self.assertIn("float(self.cfg.final_pose_success_bonus_value)", reward_source)

    def test_progress_reward_formula_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("current_potential = normalized_task_space_potential(", reward_source)
        self.assertIn("progress_raw = potential_progress_shaping(", reward_source)
        self.assertIn("reset_mask=~self._task_space_potential_valid", reward_source)

    def test_action_smoothness_penalties_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("action_rate_penalty", reward_source)
        self.assertIn("action_mag_penalty", reward_source)
        self.assertIn("-0.01", reward_source)
        self.assertIn("-0.0025", reward_source)

    def test_linear_and_angular_velocity_penalties_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("linear_velocity_penalty", reward_source)
        self.assertIn("angular_velocity_penalty", reward_source)
        self.assertIn("torch.tanh(", reward_source)

    def test_valve_sat_clamp_and_fd_penalties_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("valve_sat_penalty", reward_source)
        self.assertIn("clamp_penalty", reward_source)
        self.assertIn("fd_load_penalty", reward_source)

    def test_velocity_penalty_pose_gate_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("pose_gate", reward_source)

    def test_curriculum_log_keys_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        for key in (
            '"curriculum/stage"',
            '"curriculum/action_authority"',
            '"curriculum/episode_length"',
        ):
            self.assertIn(key, reward_source)

    def test_final_pose_log_keys_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        for key in (
            '"final_pose/position_error_mean"',
            '"final_pose/orientation_error_mean_rad"',
            '"final_pose/success_fraction"',
            '"final_pose/within_tolerance_fraction"',
        ):
            self.assertIn(key, reward_source)

    def test_adaptation_log_keys_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        for key in (
            '"adaptation/z_arm_hat_mean"',
            '"adaptation/aam_active"',
            '"adaptation/aam_frozen"',
        ):
            self.assertIn(key, reward_source)

    def test_episode_terminal_snapshots_preserved(self) -> None:
        reset_source = _method_source("_reset_idx")
        for snapshot in (
            "final_pose_success_snapshot",
            "final_position_error_snapshot",
            "final_orientation_error_snapshot",
        ):
            self.assertIn(snapshot, reset_source)


class TestUnsafePenaltyCounters(unittest.TestCase):
    """Persistent cumulative unsafe-penalty counters and terminal snapshot buffers."""

    # ── Buffer existence and dtypes (tests 1-4) ──

    def test_unsafe_penalty_event_count_total_allocated(self) -> None:
        self.assertIn("self._unsafe_penalty_event_count_total", ENV_SOURCE)
        self.assertIn("dtype=torch.long", ENV_SOURCE)
        self.assertIn("device=self.device", ENV_SOURCE)

    def test_unsafe_penalty_value_total_allocated(self) -> None:
        self.assertIn("self._unsafe_penalty_value_total", ENV_SOURCE)
        self.assertIn("dtype=torch.float32", ENV_SOURCE)

    def test_terminal_unsafe_penalty_buffer_allocated(self) -> None:
        self.assertIn("self._terminal_unsafe_penalty", ENV_SOURCE)
        self.assertIn("(self.num_envs,)", ENV_SOURCE)
        self.assertIn("dtype=torch.float32", ENV_SOURCE)

    def test_terminal_unsafe_mask_buffer_allocated(self) -> None:
        self.assertIn("self._terminal_unsafe_mask", ENV_SOURCE)
        self.assertIn("(self.num_envs,)", ENV_SOURCE)
        self.assertIn("dtype=torch.bool", ENV_SOURCE)

    # ── Counter increment logic (tests 5-11) ──

    def test_cumulative_event_counter_incremented(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("self._unsafe_penalty_event_count_total += unsafe_termination.sum().long()", reward_source)

    def test_cumulative_value_counter_incremented(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("self._unsafe_penalty_value_total += unsafe_penalty.sum()", reward_source)

    def test_terminal_unsafe_penalty_buffer_populated(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("self._terminal_unsafe_penalty[unsafe_ids]", reward_source)

    def test_terminal_unsafe_mask_buffer_populated(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("self._terminal_unsafe_mask[unsafe_ids]", reward_source)

    def test_terminal_buffer_update_uses_detached_values(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("unsafe_penalty[unsafe_ids].detach()", reward_source)
        self.assertIn("unsafe_termination[unsafe_ids].detach()", reward_source)

    def test_terminal_buffer_update_guarded_by_nonempty_ids(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn("if unsafe_ids.numel() > 0:", reward_source)

    def test_cumulative_counters_updated_before_total_reward(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        counters_index = reward_source.index("self._unsafe_penalty_event_count_total")
        total_index = reward_source.index("# ── Total reward ──")
        self.assertLess(counters_index, total_index)

    # ── Debug log keys (tests 12-13) ──

    def test_debug_log_contains_event_count_total(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn('"debug/unsafe_penalty_event_count_total"', reward_source)
        self.assertIn("self._unsafe_penalty_event_count_total.float()", reward_source)

    def test_debug_log_contains_value_total(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        self.assertIn('"debug/unsafe_penalty_value_total"', reward_source)
        self.assertIn("self._unsafe_penalty_value_total", reward_source)

    # ── Terminal snapshot capture before super (tests 14-19) ──

    def test_terminal_unsafe_penalty_snapshot_captured(self) -> None:
        reset_source = _method_source("_reset_idx")
        super_index = reset_source.index("super()._reset_idx(env_ids)")
        self.assertIn("final_unsafe_penalty_snapshot", reset_source)
        self.assertLess(
            reset_source.index("final_unsafe_penalty_snapshot"),
            super_index,
        )

    def test_terminal_unsafe_mask_snapshot_captured(self) -> None:
        reset_source = _method_source("_reset_idx")
        super_index = reset_source.index("super()._reset_idx(env_ids)")
        self.assertIn("final_unsafe_mask_snapshot", reset_source)
        self.assertLess(
            reset_source.index("final_unsafe_mask_snapshot"),
            super_index,
        )

    def test_terminal_unsafe_penalty_snapshot_from_buffer(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertIn("self._terminal_unsafe_penalty[env_ids]", reset_source)

    def test_terminal_unsafe_mask_snapshot_from_buffer(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertIn("self._terminal_unsafe_mask[env_ids]", reset_source)

    def test_terminal_unsafe_penalty_snapshot_detached_and_cloned(self) -> None:
        reset_source = _method_source("_reset_idx")
        penalty_pos = reset_source.index("final_unsafe_penalty_snapshot")
        snapshot_block = reset_source[penalty_pos:penalty_pos + 120]
        self.assertIn(".detach().clone()", snapshot_block)

    def test_terminal_unsafe_mask_snapshot_detached_and_cloned(self) -> None:
        reset_source = _method_source("_reset_idx")
        mask_pos = reset_source.index("final_unsafe_mask_snapshot")
        snapshot_block = reset_source[mask_pos:mask_pos + 120]
        self.assertIn(".detach().clone()", snapshot_block)

    # ── Terminal log key presence (tests 20-22) ──

    def test_terminal_log_contains_final_unsafe_penalty(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertIn('"Episode/final_unsafe_penalty"', reset_source)
        self.assertIn("final_unsafe_penalty_snapshot", reset_source)

    def test_terminal_log_contains_final_unsafe_mask(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertIn('"Episode/final_unsafe_mask"', reset_source)
        self.assertIn("final_unsafe_mask_snapshot", reset_source)

    def test_terminal_unsafe_mask_logged_as_float(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertIn("final_unsafe_mask_snapshot.float()", reset_source)

    # ── Post-reset buffer clearing (tests 23-24) ──

    def test_terminal_unsafe_penalty_cleared_on_reset(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertIn("self._terminal_unsafe_penalty[env_ids] = 0.0", reset_source)

    def test_terminal_unsafe_mask_cleared_on_reset(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertIn("self._terminal_unsafe_mask[env_ids] = False", reset_source)

    # ── Counter non-reset (tests 25-28) ──

    def test_cumulative_event_counter_not_reset_in_reset_idx(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertNotIn("_unsafe_penalty_event_count_total", reset_source)

    def test_cumulative_value_counter_not_reset_in_reset_idx(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertNotIn("_unsafe_penalty_value_total", reset_source)

    def test_cumulative_event_counter_not_zeroed_in_init(self) -> None:
        source_since_init = ENV_SOURCE[ENV_SOURCE.index("self._unsafe_penalty_event_count_total"):]
        self.assertNotIn("self._unsafe_penalty_event_count_total = 0", source_since_init[:300])

    def test_cumulative_value_counter_not_zeroed_in_init(self) -> None:
        source_since_init = ENV_SOURCE[ENV_SOURCE.index("self._unsafe_penalty_value_total"):]
        self.assertNotIn("self._unsafe_penalty_value_total = 0", source_since_init[:300])

    # ── Preservation of existing unsafe penalty (tests 29-35) ──

    def test_existing_unsafe_penalty_keys_preserved(self) -> None:
        reward_source = _method_source("_get_task_space_rewards")
        for key in (
            '"debug/velocity_mask_count"',
            '"debug/unsafe_mask_count"',
            '"debug/unsafe_penalty_nonzero_count"',
            '"debug/unsafe_penalty_min"',
            '"reward/unsafe_penalty_mean"',
            '"episode/unsafe_penalty_sum"',
        ):
            self.assertIn(key, reward_source)

    def test_existing_terminal_snapshot_fields_preserved(self) -> None:
        reset_source = _method_source("_reset_idx")
        for field in (
            "final_pose_success_snapshot",
            "final_position_error_snapshot",
            "final_orientation_error_snapshot",
        ):
            self.assertIn(field, reset_source)

    def test_existing_terminal_log_keys_preserved(self) -> None:
        reset_source = _method_source("_reset_idx")
        for key in (
            '"Episode/final_pose_success"',
            '"Episode/final_position_error"',
            '"Episode/final_orientation_error_rad"',
            '"Episode/final_orientation_error_deg"',
            '"Episode/final_linear_speed"',
            '"Episode/final_angular_speed"',
        ):
            self.assertIn(key, reset_source)

    def test_existing_episode_accumulators_not_affected(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertIn("self._episode_unsafe_penalty_sum[env_ids] = 0.0", reset_source)
        self.assertIn("self._episode_pos_tracking_sum[env_ids] = 0.0", reset_source)

    def test_terminal_snapshots_captured_before_super(self) -> None:
        reset_source = _method_source("_reset_idx")
        super_index = reset_source.index("super()._reset_idx(env_ids)")
        self.assertLess(
            reset_source.index("final_unsafe_penalty_snapshot"),
            super_index,
        )
        self.assertLess(
            reset_source.index("final_unsafe_mask_snapshot"),
            super_index,
        )

    def test_super_is_only_called_once_in_reset_idx(self) -> None:
        reset_source = _method_source("_reset_idx")
        self.assertEqual(reset_source.count("super()._reset_idx(env_ids)"), 1)

    def test_buffer_clearing_happens_after_episode_sums_zeroed(self) -> None:
        reset_source = _method_source("_reset_idx")
        ep_unsafe_index = reset_source.rindex("self._episode_unsafe_penalty_sum[env_ids] = 0.0")
        clear_penalty_index = reset_source.rindex("self._terminal_unsafe_penalty[env_ids] = 0.0")
        clear_mask_index = reset_source.rindex("self._terminal_unsafe_mask[env_ids] = False")
        self.assertLess(ep_unsafe_index, clear_penalty_index)
        self.assertLess(ep_unsafe_index, clear_mask_index)



class TestHoldProgressReward(unittest.TestCase):
    """Hold progress reward is positive and increasing for partial holds 1-49."""

    def test_hold_progress_reward_positive_for_partial_holds(self) -> None:
        """hold_progress_reward returns positive values for steps 1 through 49."""
        for hold_steps in range(1, 50):
            counter = torch.full((8,), hold_steps, dtype=torch.long)
            within = torch.ones(8, dtype=torch.bool)
            reward = reward_terms.hold_progress_reward(
                counter,
                required_hold_steps=50,
                weight=1.0,
                within_final_pose=within,
            )
            self.assertTrue(
                bool(torch.all(reward > 0.0)),
                f"hold_progress_reward should be positive at hold step {hold_steps}",
            )

    def test_hold_progress_reward_increases_with_hold(self) -> None:
        """hold_progress_reward increases monotonically with hold steps."""
        prev = -1.0
        for hold_steps in range(1, 50):
            counter = torch.full((8,), hold_steps, dtype=torch.long)
            within = torch.ones(8, dtype=torch.bool)
            reward = reward_terms.hold_progress_reward(
                counter,
                required_hold_steps=50,
                weight=1.0,
                within_final_pose=within,
            )
            mean_reward = reward.mean().item()
            self.assertGreater(
                mean_reward, prev,
                f"hold_progress_reward should increase from step {hold_steps - 1} to {hold_steps}",
            )
            prev = mean_reward

    def test_hold_progress_reward_zero_when_not_within_pose(self) -> None:
        """hold_progress_reward is zero when within_final_pose is False."""
        counter = torch.tensor([25, 25], dtype=torch.long)
        within = torch.tensor([False, False], dtype=torch.bool)
        reward = reward_terms.hold_progress_reward(
            counter,
            required_hold_steps=50,
            weight=1.0,
            within_final_pose=within,
        )
        self.assertTrue(bool(torch.all(reward == 0.0)))

    def test_hold_progress_reward_at_max_hold(self) -> None:
        """hold_progress_reward equals weight at exactly 50 steps."""
        counter = torch.full((4,), 50, dtype=torch.long)
        within = torch.ones(4, dtype=torch.bool)
        reward = reward_terms.hold_progress_reward(
            counter,
            required_hold_steps=50,
            weight=1.0,
            within_final_pose=within,
        )
        self.assertTrue(bool(torch.allclose(reward, torch.ones(4))))

    def test_hold_progress_reward_uses_required_hold_steps(self) -> None:
        """hold_progress_reward scales correctly with different required_hold_steps."""
        counter = torch.full((4,), 10, dtype=torch.long)
        within = torch.ones(4, dtype=torch.bool)
        reward_20 = reward_terms.hold_progress_reward(
            counter, required_hold_steps=20, weight=1.0, within_final_pose=within,
        )
        reward_50 = reward_terms.hold_progress_reward(
            counter, required_hold_steps=50, weight=1.0, within_final_pose=within,
        )
        # 10/20 = 0.5, squared = 0.25; 10/50 = 0.2, squared = 0.04
        expected_20 = (10.0 / 20.0) ** 2
        expected_50 = (10.0 / 50.0) ** 2
        self.assertAlmostEqual(reward_20[0].item(), expected_20, places=5)
        self.assertAlmostEqual(reward_50[0].item(), expected_50, places=5)


class TestSuccessBonus(unittest.TestCase):
    """Success bonus value and activation threshold are correct."""

    def test_success_bonus_value_is_10(self) -> None:
        """final_pose_success_bonus_value is exactly 10.0."""
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "final_pose_success_bonus_value",
            ),
            10.0,
        )

    def test_success_hold_steps_is_50(self) -> None:
        """final_pose_success_hold_steps is exactly 50."""
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "final_pose_success_hold_steps",
            ),
            50,
        )

    def test_success_bonus_activates_at_hold_50_not_49(self) -> None:
        """Success bonus triggers at hold >= 50, not at 49."""
        self.assertIn("final_pose_success = (", ENV_SOURCE)
        self.assertIn(
            "self._final_pose_success_counter\n            >= int(self.cfg.final_pose_success_hold_steps)",
            ENV_SOURCE,
        )


class TestSuccessThresholdsUnchanged(unittest.TestCase):
    """Strict success thresholds are unchanged."""

    def test_position_tolerance_unchanged(self) -> None:
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "final_pose_position_tolerance_m",
            ),
            0.05,
        )

    def test_orientation_tolerance_unchanged(self) -> None:
        self.assertAlmostEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "final_pose_orientation_tolerance_rad",
            ),
            0.0872664626,
            places=8,
        )

    def test_linear_speed_tolerance_unchanged(self) -> None:
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "final_pose_linear_speed_tolerance_mps",
            ),
            0.05,
        )

    def test_angular_speed_tolerance_unchanged(self) -> None:
        self.assertAlmostEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "final_pose_angular_speed_tolerance_radps",
            ),
            0.10,
        )


class TestEntropyArchitecturePreserved(unittest.TestCase):
    """Entropy / std changes preserve action dimension and model architecture."""

    def test_action_space_unchanged(self) -> None:
        """Action dimension remains 4."""
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "action_space",
            ),
            4,
        )

    def test_observation_space_unchanged(self) -> None:
        """Joint-space observation dimension is 52."""
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "observation_space",
            ),
            52,
        )

    def test_task_space_observation_dim_unchanged(self) -> None:
        """Task-space observation dimension is 70."""
        self.assertEqual(task_reference.TASK_SPACE_OBSERVATION_DIM, 70)

    def test_model_has_no_new_layers(self) -> None:
        """The LSTM model init has no new nn.Module submodules beyond original."""
        model_source = (ROOT / "standalone/workflows/skrl_ppo_lstm/lstm_model.py").read_text()
        n_linear = model_source.count("nn.Linear(")
        n_lstm = model_source.count("nn.LSTM(")
        # Original counts: 22 nn.Linear (Shared + SharedRNN classes), 1 nn.LSTM
        self.assertEqual(n_linear, 22, "nn.Linear count changed - architecture modified")
        self.assertEqual(n_lstm, 1, "nn.LSTM count changed - architecture modified")

    def test_entropy_increased_in_skrl_config(self) -> None:
        """entropy_loss_scale in skrl training is >= 0.01 (increased from 0.005)."""
        train_source = (ROOT / "standalone/workflows/skrl_ppo_lstm/mulag_train.py").read_text()
        self.assertIn('cfg["entropy_loss_scale"] = 0.02', train_source)


class TestCurriculumHysteresis(unittest.TestCase):
    """Curriculum does not advance from one isolated success."""

    def test_curriculum_advance_min_success_rate_exists(self) -> None:
        """curriculum_advance_min_success_rate config field exists with value > 0."""
        value = _literal_assignment(
            "MulagAdaptiveRLJointReachEnvCfg",
            "curriculum_advance_min_success_rate",
        )
        self.assertGreater(value, 0.0)

    def test_standard_curriculum_checks_success_rate(self) -> None:
        """Standard curriculum path references curriculum_advance_min_success_rate."""
        eval_source = _method_source("_evaluate_stage_advancement")
        self.assertIn("curriculum_advance_min_success_rate", eval_source)

    def test_hysteresis_windows_applied_in_standard_path(self) -> None:
        """Standard curriculum path uses hysteresis windows before advancing."""
        eval_source = _method_source("_evaluate_stage_advancement")
        self.assertIn("if not goal_curriculum_enabled:", eval_source)
        self.assertIn("self._curriculum_stage_ready_windows", eval_source)

    def test_goal_curriculum_hysteresis_preserved(self) -> None:
        """Goal curriculum hysteresis windows setting is preserved."""
        self.assertEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "goal_curriculum_hysteresis_windows",
            ),
            2,
        )

    def test_goal_curriculum_advance_success_rate_unchanged(self) -> None:
        """Goal curriculum advance success rate is 20%."""
        self.assertAlmostEqual(
            _literal_assignment(
                "MulagAdaptiveRLJointReachEnvCfg",
                "goal_curriculum_advance_success_rate",
            ),
            0.20,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
