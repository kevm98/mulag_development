"""Static contracts for the Kevin project-local play launcher."""

from __future__ import annotations

import ast
import copy
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLAY_PATH = ROOT / "kevin_integration/scripts/play_mulag_adaptive_rl.py"
ISAAC_PLAY_PATH = Path("/home/rrlab/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py")
SOURCE = PLAY_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


class TestPlayLauncherContract(unittest.TestCase):
    def _compatibility_namespace(self, saved_contract: dict[str, object]) -> dict[str, object]:
        helper_names = {"_space_dimension", "_validate_checkpoint_compatibility"}
        helper_source = "\n\n".join(
            ast.get_source_segment(SOURCE, node)
            for node in ast.walk(TREE)
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        )
        namespace: dict[str, object] = {
            "_agent_architecture": lambda agent_cfg: {},
            "_checkpoint_contract_payload": lambda checkpoint_path: saved_contract,
        }
        exec("from __future__ import annotations\n" + helper_source, namespace)
        return namespace

    def test_import_order_registers_kevin_before_hydra_decorator(self) -> None:
        task_import = SOURCE.index("import kevin_integration.tasks  # noqa: F401")
        verifier_call = SOURCE.index("_verify_task_registration(args_cli.task)")
        decorator = SOURCE.index("@hydra_task_config(args_cli.task, args_cli.agent)")
        self.assertLess(task_import, verifier_call)
        self.assertLess(verifier_call, decorator)
        app_launch = SOURCE.index("simulation_app = app_launcher.app")
        self.assertLess(app_launch, task_import)

    def test_invalid_task_diagnostic_contains_required_context(self) -> None:
        verifier_source = ast.get_source_segment(
            SOURCE,
            next(node for node in ast.walk(TREE) if isinstance(node, ast.FunctionDef) and node.name == "_verify_task_registration"),
        )
        self.assertIn("requested task", verifier_source)
        self.assertIn("expected registration module", verifier_source)
        self.assertIn("current working directory", verifier_source)
        self.assertIn("Python executable", verifier_source)
        self.assertIn("suggested editable-install check", verifier_source)
        self.assertIn("gym.spec(task_name)", verifier_source)

    def test_checkpoint_arguments_are_explicit_and_best_from_run_is_supported(self) -> None:
        self.assertIn('"--best_from_run"', SOURCE)
        self.assertIn("--evaluate_best_from_run", SOURCE)
        self.assertIn("if args_cli.checkpoint and args_cli.best_from_run:", SOURCE)
        self.assertIn("resume_path = retrieve_file_path(args_cli.checkpoint)", SOURCE)
        self.assertIn("_resolve_best_checkpoint(", SOURCE)

    def test_gui_mode_is_not_forced_headless(self) -> None:
        self.assertNotIn("args_cli.headless = True", SOURCE)
        self.assertNotIn("args_cli.headless=True", SOURCE)

    def test_isaaclab_source_play_is_unchanged(self) -> None:
        if not ISAAC_PLAY_PATH.is_file():
            self.skipTest("Installed IsaacLab play.py is unavailable")
        isaac_root = Path("/home/rrlab/IsaacLab")
        if not (isaac_root / ".git").is_dir():
            self.skipTest("Installed IsaacLab git checkout is unavailable")
        result = subprocess.run(
            ["git", "diff", "--", str(ISAAC_PLAY_PATH.relative_to(isaac_root))],
            cwd=isaac_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, "")

    def test_checkpoint_compatibility_checks_are_present(self) -> None:
        for token in (
            "task_name",
            "observation_dimension",
            "action_dimension",
            "runner_class",
            "policy_architecture",
            "normalizer_present",
            "environment_contract_version",
            "reward_contract_version",
        ):
            self.assertIn(token, SOURCE)

    def test_current_schema_rsl_rl_configs_do_not_use_deprecated_stochastic_migration_path(self) -> None:
        self.assertIn("def _prepare_rsl_rl_agent_cfg(", SOURCE)
        self.assertIn("def _uses_current_distribution_schema(", SOURCE)
        self.assertIn("distribution_cfg", SOURCE)
        self.assertIn("skipping deprecated stochastic-field migration", SOURCE)
        main_source = ast.get_source_segment(
            SOURCE,
            next(node for node in ast.walk(TREE) if isinstance(node, ast.FunctionDef) and node.name == "main"),
        )
        self.assertIn("_prepare_rsl_rl_agent_cfg(agent_cfg, installed_version)", main_source)

    def test_checkpoint_validation_uses_spaces_without_unwrapped_observation_call(self) -> None:
        validator_source = ast.get_source_segment(
            SOURCE,
            next(
                node
                for node in ast.walk(TREE)
                if isinstance(node, ast.FunctionDef) and node.name == "_validate_checkpoint_compatibility"
            ),
        )
        self.assertNotIn("get_observations", validator_source)
        self.assertNotIn("_get_observations", validator_source)
        self.assertIn('getattr(raw_env, "observation_space", None)', validator_source)
        self.assertIn('getattr(raw_env, "action_space", None)', validator_source)
        self.assertNotIn("env.unwrapped.get_observations", SOURCE)

    def test_direct_rl_env_without_public_get_observations_passes_space_dimension_validation(self) -> None:
        class BoxSpace:
            def __init__(self, shape):
                self.shape = shape

        class RawDirectEnv:
            observation_space = BoxSpace((70,))
            action_space = BoxSpace((4,))

            def get_observations(self):
                raise AssertionError("checkpoint compatibility must not call public observations on raw env")

        class Env:
            def __init__(self):
                self.unwrapped = RawDirectEnv()

        class EnvCfg:
            observation_space = 70
            action_space = 4
            kevin_fixed_pose_contract_version = "fixed-pose-v1"
            reward_contract_version = "reward-v1"

        class AgentCfg:
            class_name = "OnPolicyRunner"
            empirical_normalization = False

        saved_contract = {
            "task_name": "Kevin-Mulag-HydraulicROM-TaskSpace-v0",
            "observation_dimension": 70,
            "action_dimension": 4,
            "runner_class": "OnPolicyRunner",
        }
        namespace = self._compatibility_namespace(saved_contract)
        env = Env()
        before = copy.deepcopy(env.unwrapped.__dict__)
        namespace["_validate_checkpoint_compatibility"](
            "/tmp/model_1.pt",
            task_name="Kevin-Mulag-HydraulicROM-TaskSpace-v0",
            env_cfg=EnvCfg(),
            agent_cfg=AgentCfg(),
            env=env,
        )
        self.assertEqual(env.unwrapped.__dict__, before)

    def test_wrapper_get_observations_is_only_after_rsl_rl_wrapper_creation(self) -> None:
        self.assertEqual(SOURCE.count("get_observations()"), 1)
        wrapper_index = SOURCE.index("env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)")
        observations_index = SOURCE.index("obs = env.get_observations()")
        self.assertLess(wrapper_index, observations_index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
