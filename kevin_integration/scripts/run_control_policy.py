"""Starter script for verifying the Kevin integration learned control policy.

This is a minimal smoke-test harness. It does not connect to Isaac Sim.
It only confirms that the frozen learned-model chain loads and runs finite inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kevin_integration.control_policy import ControlPolicy


def assert_finite_outputs(value, prefix: str = "output") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_outputs(item, f"{prefix}.{key}")
        return

    arr = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{prefix} contains non-finite values: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="torch device string, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--steps", type=int, default=1, help="Number of finite model-inference steps to run.")
    parser.add_argument("--debug", action="store_true", help="Print the full pipeline output dictionary")
    args = parser.parse_args()

    policy = ControlPolicy(
        config={
            "pipeline": {"device": args.device},
            "return_debug": args.debug,
        }
    )

    for step in range(args.steps):
        obs = {
            "aam_x_t": np.zeros((18,), dtype=np.float32),
            "id_x_t": np.zeros((19,), dtype=np.float32),
            "fd_x_t": np.zeros((18,), dtype=np.float32),
        }
        action_or_debug = policy.compute_action(obs)
        assert_finite_outputs(action_or_debug)
        print(f"[step {step:04d}] {action_or_debug}")


if __name__ == "__main__":
    main()
