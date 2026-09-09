#!/usr/bin/env python3
"""Train PPO for the standalone PPO_AAM_FD task.

The implementation is shared with the other terrain pipelines, but this
entrypoint pins the actuator and task so a command launched from this folder
cannot silently train ROM-A or plain ID->FD.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = "RRLAB-Mulag-Verge-ID-AAM-FD-v0"


def prepare_training_args(raw_args: list[str]) -> list[str]:
    """Pin this wrapper's task and default 2--7 km/h episode-speed domain."""
    args = list(raw_args)
    for index, value in enumerate(args):
        if value == "--task":
            if index + 1 >= len(args) or args[index + 1] != TASK:
                raise SystemExit(f"{__file__} only supports --task {TASK}")
        elif value.startswith("--task=") and value.split("=", 1)[1] != TASK:
            raise SystemExit(f"{__file__} only supports --task {TASK}")
    if "--task" not in args and not any(value.startswith("--task=") for value in args):
        args += ["--task", TASK]
    has_scalar_speed = any(
        value == "--vehicle_speed_kmh"
        or value.startswith("--vehicle_speed_kmh=")
        for value in args
    )
    has_min_speed = any(
        value == "--vehicle_speed_min_kmh"
        or value.startswith("--vehicle_speed_min_kmh=")
        for value in args
    )
    has_max_speed = any(
        value == "--vehicle_speed_max_kmh"
        or value.startswith("--vehicle_speed_max_kmh=")
        for value in args
    )
    has_sampling = any(
        value == "--vehicle_speed_sampling"
        or value.startswith("--vehicle_speed_sampling=")
        for value in args
    )
    if has_min_speed != has_max_speed:
        raise SystemExit(
            "--vehicle_speed_min_kmh and --vehicle_speed_max_kmh must be given together"
        )
    # All three final pipelines share the 26 m, 2--7 km/h deployment contract.
    # A caller may replace it with one scalar or another episode-level range,
    # but the wrapper must never add a second speed mode.
    if not has_scalar_speed and not has_min_speed:
        args += [
            "--vehicle_speed_min_kmh", "2.0",
            "--vehicle_speed_max_kmh", "7.0",
        ]
    if not has_sampling:
        args += ["--vehicle_speed_sampling", "uniform"]
    return args


def main() -> int:
    args = prepare_training_args(sys.argv[1:])
    command = [
        sys.executable,
        str(ROOT / "verge_task/scripts/train.py"),
        "--actuator",
        "id_aam_fd",
        *args,
    ]
    return subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
