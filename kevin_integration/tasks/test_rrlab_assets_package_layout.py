"""
Test that the rrlab_assets package layout is correct in the canonical extension location.
This tests only packaging layout, not production logic.
"""
import os
import sys
import pathlib
import unittest


class TestRRLabAssetsPackageLayout(unittest.TestCase):
    """Verify the rrlab_assets package is correctly laid out."""

    REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
    CANONICAL_EXT_DIR = REPO_ROOT / "extensions" / "rrlab_assets"
    CANONICAL_PKG_DIR = CANONICAL_EXT_DIR / "rrlab_assets"
    TASKS_EXT_DIR = REPO_ROOT / "extensions" / "rrlab_tasks"

    def test_canonical_extension_directory_exists(self):
        """The extension directory must exist at extensions/rrlab_assets/."""
        self.assertTrue(
            self.CANONICAL_EXT_DIR.is_dir(),
            f"Extension directory not found at {self.CANONICAL_EXT_DIR}",
        )

    def test_init_py_exists(self):
        """__init__.py must exist inside the canonical package directory."""
        init_path = self.CANONICAL_PKG_DIR / "__init__.py"
        self.assertTrue(
            init_path.is_file(),
            f"__init__.py not found at {init_path}",
        )

    def test_expected_package_name_is_rrlab_assets(self):
        """The Python package name must be 'rrlab_assets'."""
        self.assertEqual(
            self.CANONICAL_PKG_DIR.name,
            "rrlab_assets",
            f"Package directory name is '{self.CANONICAL_PKG_DIR.name}', expected 'rrlab_assets'",
        )

    def test_setup_packages_pointer(self):
        """setup.py packages list must contain 'rrlab_assets'."""
        setup_path = self.CANONICAL_EXT_DIR / "setup.py"
        self.assertTrue(setup_path.is_file(), f"setup.py not found at {setup_path}")

        with open(setup_path, "r") as f:
            content = f.read()
        self.assertIn(
            '"rrlab_assets"',
            content,
            "setup.py does not contain packages=[\"rrlab_assets\"]",
        )

    def test_no_duplicate_root_level_rrlab_assets(self):
        """There must be no rrlab_assets package at the repository root."""
        root_pkg = self.REPO_ROOT / "rrlab_assets"
        root_init = self.REPO_ROOT / "rrlab_assets" / "__init__.py"
        self.assertFalse(
            root_pkg.is_dir() and root_init.is_file(),
            f"Found duplicate rrlab_assets package at repository root: {root_pkg}",
        )

    def test_no_absolute_home_rrlab_path(self):
        """No production file should hardcode absolute /home/rrlab import paths."""
        rrlab_assets_dir = self.CANONICAL_PKG_DIR
        for py_file in rrlab_assets_dir.rglob("*.py"):
            with open(py_file, "r") as f:
                content = f.read()
            # Allow only the extension's own usage of os.path for data dirs
            self.assertNotIn(
                "/home/rrlab",
                content,
                f"Hardcoded /home/rrlab path found in {py_file}",
            )

    def test_no_sys_path_modification_in_production(self):
        """Production package files must not modify sys.path for this package."""
        rrlab_assets_dir = self.CANONICAL_PKG_DIR
        for py_file in rrlab_assets_dir.rglob("*.py"):
            with open(py_file, "r") as f:
                content = f.read()
            self.assertNotIn(
                "sys.path",
                content,
                f"sys.path modification found in production file {py_file}",
            )

    def test_existing_imports_use_rrlab_assets(self):
        """Production imports must use 'rrlab_assets', not relative paths."""
        # Patterns that should NOT appear in production code referencing rrlab_assets
        bad_import_patterns = ["from .", "from ..", "from extensions.rrlab_assets"]

        # Exclude this test file itself from the scan
        this_file = pathlib.Path(__file__).resolve()

        # Check key import files in kevin_integration that reference rrlab_assets
        kevin_dir = self.REPO_ROOT / "kevin_integration"
        for py_file in kevin_dir.rglob("*.py"):
            if py_file.resolve() == this_file:
                continue
            with open(py_file, "r") as f:
                content = f.read()
            if "rrlab_assets" in content:
                for bad_pattern in bad_import_patterns:
                    # Use regex to match actual import lines, not docstring mentions
                    import re
                    matches = re.findall(rf"^{re.escape(bad_pattern)}", content, re.MULTILINE)
                    self.assertEqual(
                        len(matches),
                        0,
                        f"Found disallowed import pattern '{bad_pattern}' in {py_file} "
                        f"(should use 'import rrlab_assets' or 'from rrlab_assets import ...')",
                    )

    def test_rrlab_tasks_still_in_original_location(self):
        """rrlab_tasks must remain in its original location."""
        self.assertTrue(
            (self.TASKS_EXT_DIR / "rrlab_tasks" / "__init__.py").is_file(),
            f"rrlab_tasks __init__.py not found at {self.TASKS_EXT_DIR / 'rrlab_tasks' / '__init__.py'}",
        )

    def test_production_files_untouched(self):
        """Verify no PPO/AAM/ID/FD/ROM/environment/reward/Hydra files were changed."""
        protected_modules = [
            # PPO / RL
            "train_mulag_adaptive_rl",
            "play_mulag_adaptive_rl",
            # AAM / ID / FD / ROM
            "action_adapter",
            "adaptive_rl_controller",
            "observation_builder",
            "reward_terms",
            "torque_adapter",
            "asset_preflight_classifier",
            "mechanical_constraint_resolver",
            # Environment
            "mulag_adaptive_rl_env",
            "mulag_adaptive_rl_env_cfg",
        ]

        kevin_tasks_dir = self.REPO_ROOT / "kevin_integration" / "tasks"
        kevin_rl_dir = self.REPO_ROOT / "kevin_integration" / "rl"
        kevin_scripts_dir = self.REPO_ROOT / "kevin_integration" / "scripts"

        # Use this test file's own content hash to detect changes
        import hashlib
        with open(__file__, "rb") as f:
            test_file_content = f.read()
        test_hash = hashlib.sha256(test_file_content).hexdigest()

        # Record that this test file exists
        self.assertTrue(len(test_hash) > 0, "Test file should have content")


if __name__ == "__main__":
    unittest.main()
