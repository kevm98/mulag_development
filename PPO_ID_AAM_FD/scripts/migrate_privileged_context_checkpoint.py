#!/usr/bin/env python3
"""Expand a validated PPO checkpoint with four zero-initialized context columns.

The source checkpoint is never modified.  The new columns are normalized
context coordinates, so their initial normalizer mean is zero and std is one;
the copied actor/critic therefore produce exactly the old output for any base
observation, independent of the context value.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

CONTEXT_DIM = 4


def _expand_first_layer(state: dict[str, torch.Tensor], old_width: int, new_width: int) -> None:
    weight = state.get("mlp.0.weight")
    if weight is None or weight.ndim != 2 or int(weight.shape[1]) != int(old_width):
        raise ValueError(
            f"expected mlp.0.weight width {old_width}, got "
            f"{tuple(weight.shape) if weight is not None else None}"
        )
    expanded = torch.zeros((weight.shape[0], new_width), dtype=weight.dtype, device=weight.device)
    expanded[:, :old_width] = weight
    state["mlp.0.weight"] = expanded


def expand_checkpoint_payload(payload: dict[str, Any], *, context_dim: int = CONTEXT_DIM, source_obs_width: int | None = None) -> dict[str, Any]:
    """Return an expanded copy with optimizer state reset for safe fine-tuning."""

    if context_dim <= 0:
        raise ValueError("context_dim must be positive")
    infos = dict(payload.get("infos") or {})
    old_width = int(source_obs_width if source_obs_width is not None else infos.get("observation_width", -1))
    if old_width <= 0:
        raise ValueError("source checkpoint must declare a positive observation_width")
    new_width = old_width + int(context_dim)
    output = copy.deepcopy(payload)
    for group_name in ("actor_state_dict", "critic_state_dict"):
        state = {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in payload[group_name].items()}
        _expand_first_layer(state, old_width, new_width)
        for key, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._std", 1.0), ("obs_normalizer._var", 1.0)):
            old = state.get(key)
            if old is None or tuple(old.shape) != (1, old_width):
                raise ValueError(f"{group_name} has invalid {key} shape {getattr(old, 'shape', None)}")
            suffix = torch.full((1, context_dim), fill, dtype=old.dtype, device=old.device)
            state[key] = torch.cat((old, suffix), dim=1)
        output[group_name] = state
    output["optimizer_state_dict"] = {}
    new_infos = dict(infos)
    new_infos.update({
        "actuator": "idfd_privileged_context",
        "observation_width": new_width,
        "action_width": int(infos.get("action_width", 4)),
        "rma_context_names": [
            "actuator_authority_alpha", "hydraulic_response_time_tau",
            "actuator_delay_steps", "mass_scale",
        ][:context_dim],
        "rma_context_normalized": True,
        "initialization_source": "validated 66-D PPO ID-FD policy; appended context weights zeroed",
        "migration_requires_optimizer_reset": True,
        "optimizer_reset_on_resume": True,
    })
    output["infos"] = new_infos
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context-dim", type=int, default=CONTEXT_DIM)
    parser.add_argument("--source-obs-width", type=int, default=None)
    args = parser.parse_args()
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    output = expand_checkpoint_payload(payload, context_dim=args.context_dim, source_obs_width=args.source_obs_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "infos": output["infos"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
