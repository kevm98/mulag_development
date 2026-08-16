#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Expand a validated 66-D ID-FD policy to the 68-D AAM observation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _expand_first_layer(state: dict[str, torch.Tensor], old_width: int, new_width: int) -> None:
    weight = state.get("mlp.0.weight")
    if weight is None or tuple(weight.shape)[1] != old_width:
        raise ValueError(
            f"expected mlp.0.weight with width {old_width}, got "
            f"{tuple(weight.shape) if weight is not None else None}"
        )
    expanded = torch.zeros(
        (weight.shape[0], new_width), dtype=weight.dtype, device=weight.device
    )
    expanded[:, :old_width] = weight
    state["mlp.0.weight"] = expanded


def _read_zhat_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(path.read_text())
    if "mean" in payload and "std" in payload:
        mean = torch.as_tensor(payload["mean"], dtype=torch.float32)
        std = torch.as_tensor(payload["std"], dtype=torch.float32)
    else:
        values = [value for rows in payload["by_run"].values() for value in rows.values()]
        matrix = torch.as_tensor(values, dtype=torch.float32)
        mean = matrix.mean(dim=0)
        std = matrix.std(dim=0, unbiased=False)
    return mean.reshape(1, -1), std.reshape(1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--zhat", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    infos = dict(payload.get("infos") or {})
    if infos.get("schema") != "terrain_shared_v2":
        raise ValueError(f"unsupported source schema: {infos.get('schema')!r}")
    if infos.get("actuator") != "idfd":
        raise ValueError(f"source must be the validated ID-FD policy, got {infos.get('actuator')!r}")
    if int(infos.get("observation_width", -1)) != 66:
        raise ValueError("source policy must have the frozen 66-D ID-FD observation contract")
    if int(infos.get("action_width", -1)) != 4:
        raise ValueError("source policy must have four hydraulic actions")

    mean, std = _read_zhat_stats(args.zhat)
    if tuple(mean.shape) != (1, 2) or tuple(std.shape) != (1, 2):
        raise ValueError("--zhat must contain two-dimensional mean and std arrays")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("--zhat contains non-finite or non-positive normalization statistics")

    actor = {key: value.clone() for key, value in payload["actor_state_dict"].items()}
    critic = {key: value.clone() for key, value in payload["critic_state_dict"].items()}
    for state in (actor, critic):
        _expand_first_layer(state, old_width=66, new_width=68)
        for name, value in (
            ("obs_normalizer._mean", mean),
            ("obs_normalizer._std", std),
            ("obs_normalizer._var", std.square()),
        ):
            state[name] = torch.cat((state[name], value), dim=1)

    new_infos = dict(infos)
    new_infos.update(
        {
            "actuator": "id_aam_fd",
            "observation_width": 68,
            "action_width": 4,
            "source": "PPO3 terrain PPO",
            "initialization_source": "validated PPO4-1 ID-FD policy with zeroed AAM columns",
            "stage31_aam_normalizer_rebased": True,
            "stage31_aam_normalizer_source": str(args.zhat.resolve()),
            "stage31_aam_normalizer_mean": mean.flatten().tolist(),
            "stage31_aam_normalizer_std": std.flatten().tolist(),
            "migration_requires_optimizer_reset": True,
            "optimizer_reset_on_resume": False,
            "online_aam_residual_scale": 0.0,
        }
    )
    output = {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "optimizer_state_dict": {},
        "iter": 0,
        "infos": new_infos,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "infos": new_infos}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
