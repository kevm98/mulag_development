#!/usr/bin/env python3
"""PPO4_1 preflight: the PPO3_1 checks, run against the ID/FD plant.

Deliberately a thin wrapper rather than a copy. PPO3_1's preflight is the one
that encodes every failure this project actually hit (P4 path feasibility, S5
shaping gradients past 2x each gate, ...). A second copy would drift, and the
moment it did a PPO3_1-vs-PPO4_1 comparison would be measuring the drift.

Everything is identical except the task, which selects the actuator.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = "RRLAB-Mulag-Verge-IDFD-v0"

if __name__ == "__main__":
    env = dict(os.environ, PPO31_PREFLIGHT_TASK=TASK)
    sys.exit(
        subprocess.call(
            [
                sys.executable,
                str(ROOT / "PPO_ROM/scripts/preflight.py"),
                *sys.argv[1:],
            ],
            env=env,
            cwd=str(ROOT),
        )
    )
