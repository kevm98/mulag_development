#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Score a PPO_AAM_FD checkpoint on the shared eight-gate contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = "RRLAB-Mulag-Verge-ID-AAM-FD-v0"


def main() -> int:
    args = sys.argv[1:]
    for index, value in enumerate(args):
        if value == "--task":
            if index + 1 >= len(args) or args[index + 1] != TASK:
                raise SystemExit(f"{__file__} only supports --task {TASK}")
        elif value.startswith("--task=") and value.split("=", 1)[1] != TASK:
            raise SystemExit(f"{__file__} only supports --task {TASK}")
    if "--task" not in args and not any(value.startswith("--task=") for value in args):
        args += ["--task", TASK]
    return subprocess.call(
        [sys.executable, str(ROOT / "PPO_ROM/scripts/evaluate_gates.py"), *args],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    raise SystemExit(main())
