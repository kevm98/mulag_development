#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Construct and step the standalone PPO_AAM_FD PhysX task."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = "RRLAB-Mulag-Verge-ID-AAM-FD-v0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", default="1337")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--num_envs", type=int, default=1)
    args = parser.parse_args()
    command = [
        sys.executable,
        str(ROOT / "PPO_ROM/scripts/evaluate_gates.py"),
        "--teacher",
        "--headless",
        "--meshes",
        args.meshes,
        "--steps",
        str(args.steps),
        "--num_envs",
        str(args.num_envs),
        "--stage",
        "6",
        "--task",
        TASK,
    ]
    return subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
