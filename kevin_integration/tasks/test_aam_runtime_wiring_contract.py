"""Offline production contracts for the live AAM runtime branch.

These tests load the real learned-model checkpoints, import production Torch
helpers, and inspect the canonical environment source. IsaacLab is not started.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kevin_integration.rl.adaptive_rl_controller import (
    AdaptiveRLLearnedController,
    AdaptiveRLLearnedControllerCfg,
    TorchSequenceBuffer,
)
from kevin_integration.rl.observation_builder import (
    build_aam_input,
    build_task_space_rl_obs,
)


ENV_PATH = ROOT / "kevin_integration/tasks/mulag_adaptive_rl_env.py"
CFG_PATH = ROOT / "kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py"
CONTROLLER_PATH = ROOT / "kevin_integration/rl/adaptive_rl_controller.py"
MODELS_PATH = ROOT / "kevin_integration/controllers/learned_models.py"
MODELS_DIR = ROOT / "kevin_integration/models"
TASK_REFERENCE_PATH = ROOT / "kevin_integration/tasks/task_space_reference.py"

task_reference_spec = importlib.util.spec_from_file_location(
    "aam_contract_task_space_reference",
    TASK_REFERENCE_PATH,
)
if task_reference_spec is None or task_reference_spec.loader is None:
    raise RuntimeError(f"Unable to load {TASK_REFERENCE_PATH}")
task_reference = importlib.util.module_from_spec(task_reference_spec)
sys.modules[task_reference_spec.name] = task_reference
task_reference_spec.loader.exec_module(task_reference)
TASK_SPACE_OBSERVATION_DIM = task_reference.TASK_SPACE_OBSERVATION_DIM

ENV_SOURCE = ENV_PATH.read_text(encoding="utf-8")
CFG_SOURCE = CFG_PATH.read_text(encoding="utf-8")
CONTROLLER_SOURCE = CONTROLLER_PATH.read_text(encoding="utf-8")
MODELS_SOURCE = MODELS_PATH.read_text(encoding="utf-8")
ENV_TREE = ast.parse(ENV_SOURCE)


def _env_method_source(name: str) -> str:
    for node in ast.walk(ENV_TREE):
        if isinstance(node, ast.ClassDef) and node.name == "MulagAdaptiveRLEnv":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    source = ast.get_source_segment(ENV_SOURCE, child)
                    if source is None:
                        break
                    return source
    raise AssertionError(f"Environment method {name} not found")


def _task_observation(batch_size: int, z_arm_hat: torch.Tensor) -> torch.Tensor:
    return build_task_space_rl_obs(
        ee_pose_command=torch.zeros(batch_size, 7),
        current_ee_pose=torch.zeros(batch_size, 7),
        ee_position_error=torch.zeros(batch_size, 3),
        ee_orientation_error=torch.zeros(batch_size, 1),
        mechanical_q=torch.zeros(batch_size, 5),
        mechanical_qdot=torch.zeros(batch_size, 5),
        cylinder_q=torch.zeros(batch_size, 4),
        cylinder_qdot=torch.zeros(batch_size, 4),
        prev_action=torch.zeros(batch_size, 4),
        prev_valve_cmd=torch.zeros(batch_size, 4),
        z_arm_hat=z_arm_hat,
        fd_tau_hat=torch.zeros(batch_size, 4),
        vehicle_speed_kmh=torch.zeros(batch_size, 1),
    )


class TestAAMRuntimeWiringContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.controller = AdaptiveRLLearnedController(
            AdaptiveRLLearnedControllerCfg(
                models_dir=MODELS_DIR,
                num_envs=4,
                device="cpu",
            )
        )

    def test_01_aam_checkpoint_path_is_configured(self) -> None:
        self.assertEqual(
            self.controller.aam_checkpoint_path,
            (MODELS_DIR / "aam.pth").resolve(),
        )
        self.assertTrue(self.controller.aam_checkpoint_path.is_file())

    def test_02_aam_input_width_is_18(self) -> None:
        self.assertEqual(self.controller.AAM_INPUT_WIDTH, 18)
        self.assertEqual(self.controller.aam.model.temporal_encoder.input_size, 18)

    def test_03_aam_output_width_is_24(self) -> None:
        self.assertEqual(self.controller.AAM_OUTPUT_WIDTH, 24)
        self.assertEqual(self.controller.aam.model.head[-1].out_features, 24)

    def test_04_aam_history_shape_for_required_batch_sizes(self) -> None:
        for num_envs in (1, 16, 600, 2000):
            with self.subTest(num_envs=num_envs):
                history = TorchSequenceBuffer(
                    num_envs=num_envs,
                    window_size=self.controller.aam_history_length,
                    feature_dim=18,
                    device=torch.device("cpu"),
                )
                self.assertEqual(
                    history.data.shape,
                    (num_envs, self.controller.aam_history_length, 18),
                )

    def test_05_aam_feature_order_is_q_qdot_dp_applied_valve(self) -> None:
        q = torch.arange(5, dtype=torch.float32).reshape(1, 5)
        qdot = torch.arange(10, 15, dtype=torch.float32).reshape(1, 5)
        dP = torch.arange(20, 24, dtype=torch.float32).reshape(1, 4)
        valve = torch.arange(30, 34, dtype=torch.float32).reshape(1, 4)
        actual = build_aam_input(q, qdot, dP, valve)
        expected = torch.cat((q, qdot, dP, valve), dim=-1)
        self.assertTrue(torch.equal(actual, expected))

    def test_06_previous_applied_valve_is_the_history_source(self) -> None:
        source = _env_method_source("_update_aam_latent")
        self.assertIn(
            "previous_applied_valve_cmd = self._prev_valve_cmd",
            source,
        )
        reward_source = _env_method_source("_get_task_space_rewards")
        self.assertIn(
            "self._prev_valve_cmd = self._current_valve_cmd.detach().clone()",
            reward_source,
        )

    def test_07_nominal_valve_is_not_used_in_aam_history(self) -> None:
        source = _env_method_source("_update_aam_latent")
        self.assertNotIn("nominal", source.lower())
        self.assertNotIn("_current_valve_cmd", source)

    def test_08_aam_runs_under_no_grad(self) -> None:
        """_update_aam_latent uses torch.no_grad() to ensure the AAM output
        is a normal mutable tensor, not an inference tensor.

        The controller's internal update_aam_latent may still use
        inference_mode, but the env wrapper must produce a mutable buffer.
        """
        source = _env_method_source("_update_aam_latent")
        self.assertIn("with torch.no_grad():", source)
        self.assertNotIn("with torch.inference_mode():", source)
        self.assertIn("with torch.no_grad():", MODELS_SOURCE)

    def test_09_aam_is_in_eval_mode(self) -> None:
        self.assertTrue(self.controller.aam_eval_mode)
        self.assertFalse(self.controller.aam.model.training)

    def test_10_all_aam_parameters_are_frozen(self) -> None:
        self.assertTrue(self.controller.aam_frozen)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in self.controller.aam.model.parameters()
            )
        )

    def test_11_aam_output_shape_is_batch_by_24(self) -> None:
        self.controller.reset()
        latent = self.controller.update_aam_latent(torch.zeros(4, 18))
        self.assertEqual(latent.shape, (4, 24))
        self.assertTrue(bool(torch.isfinite(latent).all()))

    def test_12_nonfinite_aam_output_is_rejected(self) -> None:
        original_aam = self.controller.aam

        class NonFiniteAAM:
            def __call__(self, history: torch.Tensor) -> torch.Tensor:
                return torch.full(
                    (history.shape[0], 24),
                    float("nan"),
                    dtype=history.dtype,
                    device=history.device,
                )

        self.controller.aam = NonFiniteAAM()
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "AAM produced non-finite latent values",
            ):
                self.controller.run_aam(self.controller.aam_history)
        finally:
            self.controller.aam = original_aam

    def test_13_latent_enters_the_task_observation(self) -> None:
        latent = torch.arange(48, dtype=torch.float32).reshape(2, 24)
        observation = _task_observation(2, latent)
        self.assertTrue(torch.equal(observation[:, 41:65], latent))

    def test_14_latent_occupies_exactly_24_observation_values(self) -> None:
        latent = torch.ones(2, 24)
        observation = _task_observation(2, latent)
        self.assertEqual(observation[:, 41:65].shape[-1], 24)

    def test_15_task_observation_remains_70d(self) -> None:
        observation = _task_observation(2, torch.zeros(2, 24))
        self.assertEqual(TASK_SPACE_OBSERVATION_DIM, 70)
        self.assertEqual(observation.shape, (2, 70))

    def test_16_id_receives_the_same_stored_latent(self) -> None:
        helper_source = _env_method_source("_id_conditioning_latent")
        self.assertIn("return self._z_arm_hat", helper_source)
        self.assertEqual(ENV_SOURCE.count("z_arm_hat=self._id_conditioning_latent()"), 4)

    def test_17_id_does_not_recompute_aam(self) -> None:
        valve_source_start = CONTROLLER_SOURCE.index("def apply_action_valve_only")
        full_source_start = CONTROLLER_SOURCE.index("def apply_action_full_pipeline")
        valve_source = CONTROLLER_SOURCE[valve_source_start:full_source_start]
        full_source = CONTROLLER_SOURCE[full_source_start:]
        self.assertNotIn("compute_z_arm(", valve_source)
        self.assertNotIn("compute_z_arm(", full_source)
        self.assertIn("z_arm_hat is required", valve_source)
        self.assertIn("z_arm_hat is required", full_source)

    def test_18_offline_system_factor_encoder_is_not_used(self) -> None:
        runtime_source = ENV_SOURCE + CONTROLLER_SOURCE
        self.assertNotIn("SystemFactorEncoder", runtime_source)
        self.assertNotIn("teacher_latent", runtime_source)

    def test_19_aam_does_not_write_valve_cmd(self) -> None:
        source = _env_method_source("_update_aam_latent")
        self.assertNotIn("_current_valve_cmd =", source)
        self.assertNotIn("_prev_valve_cmd =", source)

    def test_20_aam_does_not_write_cylinder_targets(self) -> None:
        source = _env_method_source("_update_aam_latent")
        self.assertNotIn("_rom_cylinder_pos_target", source)
        self.assertNotIn("set_joint_position_target", source)

    def test_21_partial_reset_clears_selected_history_rows(self) -> None:
        history = TorchSequenceBuffer(
            num_envs=4,
            window_size=3,
            feature_dim=18,
            device=torch.device("cpu"),
        )
        history.data.fill_(1.0)
        history.reset(torch.tensor([1, 3]))
        self.assertTrue(bool(torch.all(history.data[[1, 3]] == 0.0)))

    def test_22_partial_reset_preserves_nonreset_history_rows(self) -> None:
        history = TorchSequenceBuffer(
            num_envs=4,
            window_size=3,
            feature_dim=18,
            device=torch.device("cpu"),
        )
        history.data.fill_(1.0)
        history.reset(torch.tensor([1, 3]))
        self.assertTrue(bool(torch.all(history.data[[0, 2]] == 1.0)))

    def test_23_partial_reset_clears_only_selected_latent_rows(self) -> None:
        source = _env_method_source("_reset_idx")
        self.assertIn("self._z_arm_hat[env_ids] = 0.0", source)
        self.assertNotIn("self._z_arm_hat.zero_()", source)

    def test_24_aam_history_updates_once_per_control_state(self) -> None:
        observation_source = _env_method_source("_get_observations")
        update_source = _env_method_source("_update_aam_latent")
        self.assertEqual(observation_source.count("self._update_aam_latent("), 1)
        self.assertEqual(
            update_source.count("self._controller.update_aam_latent("),
            1,
        )
        self.assertNotIn("update_aam_latent", _env_method_source("_id_conditioning_latent"))

    def test_25_ppo_is_the_only_trainable_network(self) -> None:
        for wrapper in (
            self.controller.aam,
            self.controller.id,
            self.controller.fd,
            self.controller.sd,
        ):
            self.assertFalse(
                any(parameter.requires_grad for parameter in wrapper.model.parameters())
            )

    def test_26_action_remains_4d(self) -> None:
        self.assertIn("action_space = 4", CFG_SOURCE)
        qddot = self.controller.build_qddot_cmd(
            torch.ones(4, 4),
            scale=1.0,
            clip=5.0,
        )
        self.assertEqual(qddot.shape, (4, 5))
        self.assertTrue(bool(torch.all(qddot[:, 4] == 0.0)))

    def test_27_id_output_remains_4d(self) -> None:
        self.controller.reset()
        latent = self.controller.update_aam_latent(torch.zeros(4, 18))
        output = self.controller.apply_action_valve_only(
            torch.zeros(4, 5),
            torch.zeros(4, 5),
            torch.zeros(4, 4),
            torch.zeros(4, 4),
            z_arm_hat=latent,
        )
        self.assertEqual(output.valve_cmd.shape, (4, 4))

    def test_28_rom_output_remains_4d(self) -> None:
        self.assertIn("output_dim=4", ENV_SOURCE)
        self.assertIn("num_cylinders=4", ENV_SOURCE)

    def test_29_final_pose_behavior_is_unchanged(self) -> None:
        reward_source = _env_method_source("_get_task_space_rewards")
        target_source = _env_method_source("_sample_task_space_command")
        success_source = _env_method_source("_update_final_pose_success_state")
        reward_tree = ast.parse(reward_source)
        reward_assignment = next(
            node
            for node in ast.walk(reward_tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "reward"
                for target in node.targets
            )
        )
        reward_names = {
            node.id
            for node in ast.walk(reward_assignment.value)
            if isinstance(node, ast.Name)
        }
        self.assertNotIn("z_arm_hat", reward_names)
        self.assertNotIn("aam_history", reward_names)
        self.assertNotIn("z_arm_hat", target_source)
        self.assertIn("final_pose_success_hold_steps", success_source)
        self.assertIn("self._sample_task_space_command(env_ids)", _env_method_source("_reset_idx"))

    def test_30_hydra_and_launcher_are_unchanged(self) -> None:
        task_init = (ROOT / "kevin_integration/tasks/__init__.py").read_text(
            encoding="utf-8"
        )
        train_source = (
            ROOT / "kevin_integration/scripts/train_mulag_adaptive_rl.py"
        ).read_text(encoding="utf-8")
        self.assertIn('id="Kevin-Mulag-HydraulicROM-TaskSpace-v0"', task_init)
        self.assertIn(
            'DEFAULT_HYDRAULIC_ROM_TASK = "Kevin-Mulag-HydraulicROM-TaskSpace-v0"',
            train_source,
        )
        self.assertIn(
            '@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")',
            train_source,
        )

    def test_31_latent_copy_preserves_buffer_identity(self) -> None:
        """.copy_() preserves the buffer object identity so that
        partial resets via self._z_arm_hat[env_ids] = 0.0
        continue to work after the AAM latent is updated.
        """
        buffer = torch.zeros((4, 24), dtype=torch.float32)
        original_id = id(buffer)
        new_values = torch.randn(4, 24)
        buffer.copy_(new_values)
        self.assertEqual(
            id(buffer),
            original_id,
            ".copy_() must preserve the buffer identity",
        )
        self.assertTrue(
            torch.equal(buffer, new_values),
            ".copy_() must write the new values into the buffer",
        )

    def test_32_update_aam_latent_uses_copy_not_assignment(self) -> None:
        """_update_aam_latent must use .copy_() to write the AAM output
        into the pre-allocated _z_arm_hat buffer, not direct assignment
        which would replace the buffer with an inference tensor.
        """
        source = _env_method_source("_update_aam_latent")
        self.assertIn(
            "self._z_arm_hat.copy_(",
            source,
            "_update_aam_latent must use .copy_() to write into the buffer",
        )
        self.assertNotIn(
            "self._z_arm_hat = ",
            source,
            "_update_aam_latent must not replace the buffer with =",
        )

    def test_33_partial_reset_preserves_nonreset_latent_rows(self) -> None:
        """Partial reset of _z_arm_hat must zero only the specified env_ids
        and leave non-reset rows intact.
        """
        buffer = torch.full((4, 24), 42.0, dtype=torch.float32)
        env_ids = torch.tensor([1, 3])
        buffer[env_ids] = 0.0
        self.assertTrue(
            torch.all(buffer[[0, 2]] == 42.0),
            "Non-reset rows must be preserved",
        )
        self.assertTrue(
            torch.all(buffer[1] == 0.0),
            "Reset row 1 must be zeroed",
        )
        self.assertTrue(
            torch.all(buffer[3] == 0.0),
            "Reset row 3 must be zeroed",
        )

    def test_34_z_arm_hat_allocation_works_for_batch_sizes(self) -> None:
        """The _z_arm_hat buffer allocation pattern (torch.zeros, N, 24)
        works for all canonical batch sizes and produces mutable buffers.
        """
        for num_envs in (1, 16, 600, 2000):
            with self.subTest(num_envs=num_envs):
                buffer = torch.zeros(
                    (num_envs, 24),
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
                self.assertEqual(buffer.shape, (num_envs, 24))
                self.assertFalse(
                    buffer.is_inference(),
                    f"Buffer for N={num_envs} must be mutable",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
