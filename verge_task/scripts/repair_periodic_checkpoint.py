#!/usr/bin/env python3
"""Restore provenance on an old RSL-RL interval checkpoint.

New training runs wrap every save with provenance. This utility exists only
for interval files written before that fix, especially after interruption.
It never overwrites its input and does not alter model or optimizer tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("periodic", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run_manifest", type=Path)
    args = parser.parse_args()

    periodic = args.periodic.resolve()
    reference = args.reference.resolve()
    output = args.output.resolve()
    if output in {periodic, reference}:
        parser.error("output must not overwrite the periodic or reference checkpoint")

    payload = torch.load(periodic, map_location="cpu", weights_only=False)
    baseline = torch.load(reference, map_location="cpu", weights_only=False)
    for key in ("actor_state_dict", "critic_state_dict"):
        actual = payload.get(key) or {}
        expected = baseline.get(key) or {}
        if actual.keys() != expected.keys() or any(
            actual[name].shape != expected[name].shape for name in expected
        ):
            raise RuntimeError(f"{key} contract differs from reference")

    infos = dict(baseline.get("infos") or {})
    infos.pop("migration_requires_optimizer_reset", None)
    if args.run_manifest:
        manifest = json.loads(args.run_manifest.resolve().read_text())
        for key in (
            "actuator",
            "initialization_source",
            "seed",
            "terrain_base_seed",
            "terrain_curriculum_version",
            "terrain_height_scales",
            "path_families",
            "head_joint_action",
            "head_action_to_c4_gain",
            "observation_width",
            "action_width",
            "tail_focus",
            "lateral_focus",
        ):
            if key in manifest:
                infos[key] = manifest[key]
    infos.update(
        {
            "source": "PPO3 terrain PPO",
            "schema": "terrain_shared_v2",
            "iteration": int(payload.get("iter", -1)),
            "recovered_periodic_checkpoint": True,
            "recovery_reference": str(reference),
        }
    )
    payload["infos"] = infos
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(output)


if __name__ == "__main__":
    main()
