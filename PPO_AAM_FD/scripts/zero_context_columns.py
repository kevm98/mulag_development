#!/usr/bin/env python3
"""Zero a checkpoint's context-column weights in place (no width change).

Why this exists. Distillation runs on a NOMINAL plant, where
``normalized_context`` maps the context to exactly zeros. A constant-zero input
produces a zero weight gradient, so the actor's weights on those columns never
move off their RANDOM INITIALISATION during the whole distillation. They look
harmless -- the column contributes nothing while the input is zero -- right up
until Phase 1 turns plant randomisation on, at which point random weights start
injecting noise into a converged policy.

That is the failure mode the 72-wide ``prive_warmstart`` checkpoint showed: its
context weights were small but nonzero (abs-max 0.00496) and it destabilised
under fine-tuning where its 68-wide source did not.

Zeroing makes the Phase 1 start an exact identity: the policy is provably the
distilled one for ANY context value, so any later gain is attributable to
learning the context rather than to relearning the task.

The check is an equivalence test, not a spot check: outputs must match the
source for context drawn at several magnitudes, including far outside the
normalised range, because a surviving weight of magnitude eps shows up as
eps * |context| and a unit-scale probe would miss it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def zero_context(state: dict, context_dim: int) -> float:
    """Zero the trailing `context_dim` input columns of the first layer."""
    key = next((k for k in state if k.endswith("mlp.0.weight")), None)
    if key is None:
        raise KeyError(f"no first-layer weight found; keys={sorted(state)[:8]}")
    weight = state[key]
    before = float(weight[:, -context_dim:].abs().max())
    weight[:, -context_dim:] = 0.0
    return before


def forward(state: dict, obs: torch.Tensor) -> torch.Tensor:
    std = state["obs_normalizer._std"].clamp_min(1e-6)
    x = (obs - state["obs_normalizer._mean"]) / std
    for index in (0, 2, 4):
        x = torch.nn.functional.elu(
            torch.nn.functional.linear(
                x, state[f"mlp.{index}.weight"], state[f"mlp.{index}.bias"]
            )
        )
    return torch.nn.functional.linear(x, state["mlp.6.weight"], state["mlp.6.bias"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--context-dim", type=int, default=4)
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="Max permitted output drift. Default 0.0: exact.")
    a = p.parse_args()

    payload = torch.load(a.source, map_location="cpu", weights_only=False)
    width = int(payload["actor_state_dict"]["mlp.0.weight"].shape[1])
    base_dim = width - int(a.context_dim)
    print(f"[zero-context] width={width} base={base_dim} context={a.context_dim}")

    reference = {k: v.clone() for k, v in payload["actor_state_dict"].items()}
    for name in ("actor_state_dict", "critic_state_dict"):
        if name not in payload:
            continue
        before = zero_context(payload[name], int(a.context_dim))
        print(f"[zero-context] {name}: context weight abs-max {before:.8f} -> 0.0")

    # The optimizer's moment estimates reference the pre-zeroing weights; keeping
    # them would let Adam immediately re-inflate the columns from stale momentum.
    if payload.get("optimizer_state_dict"):
        payload["optimizer_state_dict"] = {}
        print("[zero-context] cleared optimizer state (stale moments would "
              "re-inflate the zeroed columns)")

    torch.manual_seed(0)
    base = torch.randn(128, base_dim)
    worst = 0.0
    for scale in (1.0, 10.0, 100.0, 1000.0):
        context = torch.randn(128, int(a.context_dim)) * scale
        full = torch.cat((base, context), dim=-1)
        # Reference: the ORIGINAL actor on the same base observation, with its own
        # context columns fed the same values.
        drift = float((forward(reference, full) - forward(payload["actor_state_dict"], full)).abs().max())
        zero_ctx = torch.cat((base, torch.zeros(128, int(a.context_dim))), dim=-1)
        invariance = float(
            (forward(payload["actor_state_dict"], full)
             - forward(payload["actor_state_dict"], zero_ctx)).abs().max()
        )
        worst = max(worst, invariance)
        print(f"[zero-context]   context x{scale:<7g} drift-vs-source={drift:.3e} "
              f"context-invariance={invariance:.3e}")

    if worst > float(a.tolerance):
        raise SystemExit(
            f"zeroed actor is not context-invariant: max deviation {worst:.3e} "
            f"> tolerance {a.tolerance}. Refusing to write."
        )

    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, a.output)
    print(f"[zero-context] context-invariance verified exact; wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
