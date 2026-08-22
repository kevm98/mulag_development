#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Create an audited actor interpolation between compatible PPO checkpoints."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")

    base_path = args.base.expanduser().resolve()
    target_path = args.target.expanduser().resolve()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    target = torch.load(target_path, map_location="cpu", weights_only=False)
    base_actor = base["actor_state_dict"]
    target_actor = target["actor_state_dict"]
    if set(base_actor) != set(target_actor):
        raise RuntimeError("actor state dictionaries have different keys")
    if (base.get("infos") or {}).get("observation_width") != (
        target.get("infos") or {}
    ).get("observation_width"):
        raise RuntimeError("checkpoint observation contracts differ")

    blended = {}
    alpha = float(args.alpha)
    for name, base_value in base_actor.items():
        target_value = target_actor[name]
        if base_value.shape != target_value.shape or base_value.dtype != target_value.dtype:
            raise RuntimeError(f"incompatible actor tensor {name!r}")
        if torch.is_floating_point(base_value):
            blended[name] = (1.0 - alpha) * base_value + alpha * target_value
        else:
            if not torch.equal(base_value, target_value):
                raise RuntimeError(f"non-floating actor state differs for {name!r}")
            blended[name] = base_value.clone()

    payload = deepcopy(base)
    payload["actor_state_dict"] = blended
    # The inherited optimizer moments do not correspond to interpolated actor
    # weights. Evaluation does not load them; any later resume must explicitly
    # reset the optimizer, which this flag enforces in train.py.
    payload["optimizer_state_dict"] = {}
    infos = dict(payload.get("infos") or {})
    infos.update(
        {
            "checkpoint_kind": "audited_actor_weight_interpolation",
            "actor_blend_base": str(base_path),
            "actor_blend_target": str(target_path),
            "actor_blend_alpha": alpha,
            "migration_requires_optimizer_reset": True,
        }
    )
    payload["infos"] = infos
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(
        f"wrote {args.out} alpha={alpha:g} "
        f"base={base_path.name} target={target_path.name}"
    )


if __name__ == "__main__":
    main()
