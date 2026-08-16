#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Create a 68-observation teacher-student initializer for PPO_ID_AAM_FD."""

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
    command = [
        sys.executable,
        str(ROOT / "verge_task/scripts/distill_terrain.py"),
        "--actuator",
        "id_aam_fd",
        *args,
    ]
    return subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
