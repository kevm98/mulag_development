"""Static contracts for the Kevin project-local play launcher."""

from __future__ import annotations

import ast
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
