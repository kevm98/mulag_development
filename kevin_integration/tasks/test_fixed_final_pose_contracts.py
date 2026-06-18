"""Compatibility entrypoint for fixed final-pose contract tests."""

from __future__ import annotations

import importlib.util
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_TEST_PATH = Path(__file__).with_name("test_fixed_final_pose_task.py")
spec = importlib.util.spec_from_file_location("fixed_final_pose_task_contracts", TASK_TEST_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {TASK_TEST_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

for name, value in vars(module).items():
    if name.startswith("Test"):
        globals()[name] = value


if __name__ == "__main__":
    unittest.main(verbosity=2)
