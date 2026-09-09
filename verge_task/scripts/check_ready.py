#!/usr/bin/env python3
"""Check that a PPO3 checkpoint matches the shared 66x5 ROM contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default=str(DEFAULT))
    args = parser.parse_args()
    path = Path(args.checkpoint).expanduser().resolve()
    if not path.is_file():
        print(f"[FAIL] checkpoint missing: {path}")
        return 1

    payload = torch.load(path, map_location="cpu", weights_only=False)
    infos = payload.get("infos") or {}
    actor = payload.get("actor_state_dict") or {}
    checks = {
        "schema terrain_shared_v2": infos.get("schema") == "terrain_shared_v2",
        "actuator rom": str(infos.get("actuator", "rom")) == "rom",
        "observation width 66": tuple(actor["mlp.0.weight"].shape)[1] == 66,
        "action width 5": tuple(actor["mlp.6.weight"].shape)[0] == 5,
        "normalizer width 66": tuple(actor["obs_normalizer._mean"].shape) == (1, 66),
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if all(checks.values()):
        print(f"READY: ./verge_task/visualize.sh {path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
