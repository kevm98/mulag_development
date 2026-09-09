#!/usr/bin/env python3
"""Show the newest PPO_AAM_FD TensorBoard/checkpoint run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "logs/rsl_rl/ppo_id_aam_fd"


def main() -> int:
    args = sys.argv[1:]
    if "--run" not in args:
        candidates = [
            path
            for path in DEFAULT_ROOT.glob("*")
            if path.is_dir() and list(path.glob("events.out.tfevents*"))
        ]
        if not candidates:
            raise SystemExit(
                f"no PPO_AAM_FD TensorBoard run found under {DEFAULT_ROOT}; "
                "pass --run PATH explicitly"
            )
        args += ["--run", str(max(candidates, key=lambda path: path.stat().st_mtime))]
    return subprocess.call(
        [sys.executable, str(ROOT / "PPO_ROM/scripts/training_status.py"), *args],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    raise SystemExit(main())
