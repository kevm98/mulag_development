"""Report how much of a checkpoint's actor input layer sits on the context.

The mechanism probe for the payload-conditioning campaign, and it comes BEFORE
the outcome (memory.md invariant 9). The frozen base carries mean |w| = 0.125463
on the 66 task columns and exactly 0.000000 on the four context columns, so an
adaptive arm whose context weights are still ~0 after training cannot differ
from its blind twin no matter what the scoring says. Reading this during
training catches a null hours before the gate does.

Column order is CONTEXT_NAMES from PPO_AAM_FD.rma_adaptation: alpha, tau, delay,
mass_scale. Only mass_scale varies in this campaign; the other three are
constant, so their weights carry no gradient and are expected to stay at zero.
That expectation is itself a check: if alpha/tau/delay move as much as
mass_scale, the run is fitting noise rather than the payload.

Usage:  ctxweights.py <checkpoint.pt> [more.pt ...]
"""
import sys

import torch


def report(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") or payload.get("actor_state_dict") or payload
    if not isinstance(state, dict):
        state = payload
    for key, value in state.items():
        if not (hasattr(value, "ndim") and value.ndim == 2):
            continue
        if "normalizer" in key or value.shape[1] < 68:
            continue
        width = value.shape[1]
        n_ctx = 4 if width == 70 else (6 if width == 72 else 2)
        base = value[:, : width - n_ctx].abs().mean().item()
        ctx = value[:, width - n_ctx :].abs().mean().item()
        names = ("alpha", "tau", "delay", "mass_scale")
        cols = " ".join(
            f"{names[i] if i < len(names) else i}={value[:, width - n_ctx + i].abs().mean().item():.6f}"
            for i in range(n_ctx)
        )
        print(f"{path}")
        print(f"  {key} {tuple(value.shape)}  base={base:.6f} ctx={ctx:.6f} "
              f"ratio={ctx / base if base else float('nan'):.5f}")
        print(f"    {cols}")
        return
    print(f"{path}: no context-carrying layer found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        report(path)
