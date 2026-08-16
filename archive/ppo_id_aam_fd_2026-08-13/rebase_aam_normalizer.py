#!/usr/bin/env python3
"""Rebase only the PPO observation-normalizer channels for Stage 31 z_hat.

The actor and critic weights stay unchanged. This is needed when a frozen
68-D checkpoint was trained with an older AAM latent coordinate system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _latent_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    payload = json.loads(path.read_text())
    values = []
    for run in (payload.get("by_run") or {}).values():
        values.extend(run.values())
    z = torch.as_tensor(values, dtype=torch.float32)
    if z.ndim != 2 or z.shape[1] != 2 or z.shape[0] < 100:
        raise ValueError(f"expected >=100 two-dimensional z_hat rows, got {tuple(z.shape)}")
    return z.mean(dim=0), z.std(dim=0, unbiased=True).clamp_min(1.0e-6), int(z.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("zhat_json")
    parser.add_argument("output")
    args = parser.parse_args()

    source = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    mean, std, count = _latent_stats(Path(args.zhat_json).expanduser().resolve())
    payload = torch.load(source, map_location="cpu", weights_only=False)
    infos = dict(payload.get("infos") or {})
    if int(infos.get("observation_width", -1)) != 68:
        raise ValueError("Stage 31 AAM normalizer migration requires a 68-D checkpoint")

    for name in ("actor_state_dict", "critic_state_dict"):
        state = payload[name]
        for key in ("obs_normalizer._mean", "obs_normalizer._std", "obs_normalizer._var"):
            tensor = state[key]
            if tuple(tensor.shape) != (1, 68):
                raise ValueError(f"{name}.{key} has unexpected shape {tuple(tensor.shape)}")
        state["obs_normalizer._mean"][0, -2:] = mean
        state["obs_normalizer._std"][0, -2:] = std
        state["obs_normalizer._var"][0, -2:] = std.square()

    infos.update(
        {
            "stage31_aam_normalizer_rebased": True,
            "stage31_aam_normalizer_source": str(Path(args.zhat_json).expanduser().resolve()),
            "stage31_aam_normalizer_rows": count,
            "migration_requires_optimizer_reset": True,
        }
    )
    payload["infos"] = infos
    payload.pop("optimizer_state_dict", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        f"rebased {source} -> {output}; z_hat mean={mean.tolist()} std={std.tolist()} "
        f"rows={count}; optimizer state removed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
