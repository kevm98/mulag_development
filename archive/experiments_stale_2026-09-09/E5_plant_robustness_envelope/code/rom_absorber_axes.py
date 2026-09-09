#!/usr/bin/env python
"""E5 Stage 2: the two absorbers that sit between a plant change and the plant.

CPU-only, no Isaac. Stage 1 ranked which physical parameters can degrade the
ROM, but every one of those numbers is conditional on two constants nobody has
swept:

1. **The direction guard.** `command_rom_direction_guard` clamps realised motion
   to `command_rom_command_max_ratio x |command|` (3.0 in the deployment configs)
   and zeros any wrong-direction response. Stage 1 showed it REORDERS which
   factor is most damaging rather than attenuating uniformly, so it cannot be
   treated as a constant gain.
2. **The learned residual heads.** They run in parallel with the physical
   pressure path and carry no dependence on any hydraulic parameter. At nominal
   they supply ~31% of the extension increment, so roughly a third of the ROM's
   motion is by construction undegradable through physics.

This screen makes both into axes and measures TRANSMISSION: how much of a fixed
plant perturbation survives to the realised extension as each absorber is
varied. The reference stress is leakage at 0.5x -- Stage 1's strongest axis,
taken inside its clean region (dP saturation 30%, well clear of the clamp that
pins the 0.1x and 0.01x points at 100%).

Prediction under test: as the residual gain falls to zero the ROM becomes pure
grey-box physics, so the SAME hydraulic perturbation should bite HARDER. If it
does not, the physics path is not what carries the motion at all.

Residual gain
-------------
`r` scales the final Linear layer of every residual head. Because each head's
output passes through a tanh, this shrinks the pre-tanh drive: r=1 is exactly
nominal, r=0 is exactly zero residual, and the interpolation between is monotone
but not linear. That is sufficient for an absorber sweep and requires no edit to
the model.

Usage:
    python3.11 experiments/E5_plant_robustness_envelope/code/rom_absorber_axes.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rom_stress_lib import (  # noqa: E402
    COMMAND_MAX_RATIO, ROOT, TRAJECTORIES, OutputTap, build_runtime,
    get_param, metrics, rollout, set_param,
)

GUARD_RATIOS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0)
RESIDUAL_GAINS = (0.0, 0.25, 0.5, 0.75, 1.0)
REFERENCE_STRESS = ("log_leakage", 0.5)   # Stage 1's strongest axis, clean region

RESIDUAL_HEADS = (
    "pressure_residual_head", "velocity_residual_head", "position_residual_head",
    "force_residual_head", "load_head", "command_extension_correction_head",
)


def final_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    return module[-1]


@contextlib.contextmanager
def residual_gain(model, gain: float):
    """Temporarily scale every residual head's output layer."""
    saved = {}
    for name in RESIDUAL_HEADS:
        layer = final_linear(getattr(model, name))
        saved[name] = (layer.weight.detach().clone(), layer.bias.detach().clone())
        with torch.no_grad():
            layer.weight.mul_(gain)
            layer.bias.mul_(gain)
    try:
        yield
    finally:
        for name, (w, b) in saved.items():
            layer = final_linear(getattr(model, name))
            with torch.no_grad():
                layer.weight.copy_(w)
                layer.bias.copy_(b)


def transmission(runtime, command, limits, *, guard, guard_ratio, tap, beta_nom_key,
                 nominal_value, stress_scale):
    """Nominal and stressed rollouts at one absorber setting; how much survives."""
    model = runtime.model
    set_param(model, beta_nom_key, nominal_value)
    nom = rollout(runtime, command, limits, guard=guard, tap=tap, guard_ratio=guard_ratio)
    set_param(model, beta_nom_key, nominal_value * stress_scale)
    stressed = rollout(runtime, command, limits, guard=guard, tap=tap, guard_ratio=guard_ratio)
    set_param(model, beta_nom_key, nominal_value)
    return metrics(nom, nom), metrics(nom, stressed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"))
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(ROOT / "experiments/E5_plant_robustness_envelope/results/rom_absorber_axes.json"))
    args = p.parse_args()

    torch.manual_seed(17)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    runtime = build_runtime(args.checkpoint, device)
    model = runtime.model
    limits = runtime.checkpoint["cylinder_limits"].to(device)
    command = TRAJECTORIES["all_axes"](args.ticks, device)
    tap = OutputTap(model)

    key, stress_scale = REFERENCE_STRESS
    nominal_value = get_param(model, key)
    print(f"checkpoint      : {args.checkpoint}")
    print(f"probe           : all_axes, {args.ticks} ticks, guard ratio default {COMMAND_MAX_RATIO}")
    print(f"reference stress: {key} x{stress_scale}\n")

    payload: dict = {
        "checkpoint": args.checkpoint, "ticks": args.ticks,
        "reference_stress": {"parameter": key, "scale": stress_scale},
        "guard_ratios": list(GUARD_RATIOS), "residual_gains": list(RESIDUAL_GAINS),
    }

    # ---- axis 1: the direction guard ---------------------------------------
    print("AXIS 1 -- direction guard ratio (deployment value is 3.0)")
    print(f"{'ratio':>8}{'travel mm':>11}{'authority':>11}{'veto':>8}"
          f"{'transmitted':>13}{'err mm':>9}")
    guard_rows = []
    for ratio in GUARD_RATIOS:
        nom_m, str_m = transmission(
            runtime, command, limits, guard=True, guard_ratio=ratio, tap=tap,
            beta_nom_key=key, nominal_value=nominal_value, stress_scale=stress_scale)
        guard_rows.append({"guard_ratio": ratio, "nominal": nom_m, "stressed": str_m})
        print(f"{ratio:>8.2f}{nom_m['travel_mm']:>11.3f}{nom_m['authority_kept']*100:>10.1f}%"
              f"{nom_m['guard_veto_rate']*100:>7.1f}%"
              f"{str_m['rel_peak_extension']*100:>12.1f}%{str_m['peak_extension_err_mm']:>9.3f}")
    nom_m, str_m = transmission(
        runtime, command, limits, guard=False, guard_ratio=None, tap=tap,
        beta_nom_key=key, nominal_value=nominal_value, stress_scale=stress_scale)
    guard_rows.append({"guard_ratio": None, "nominal": nom_m, "stressed": str_m})
    print(f"{'off':>8}{nom_m['travel_mm']:>11.3f}{nom_m['authority_kept']*100:>10.1f}%"
          f"{nom_m['guard_veto_rate']*100:>7.1f}%"
          f"{str_m['rel_peak_extension']*100:>12.1f}%{str_m['peak_extension_err_mm']:>9.3f}")
    payload["guard_axis"] = guard_rows

    # ---- axis 2: the learned residual heads --------------------------------
    print("\nAXIS 2 -- residual gain (1.0 = the trained ROM, 0.0 = pure grey-box)")
    print(f"{'gain':>8}{'travel mm':>11}{'residual dx':>13}{'transmitted':>13}{'err mm':>9}")
    residual_rows = []
    for gain in RESIDUAL_GAINS:
        with residual_gain(model, gain):
            nom_m, str_m = transmission(
                runtime, command, limits, guard=True, guard_ratio=None, tap=tap,
                beta_nom_key=key, nominal_value=nominal_value, stress_scale=stress_scale)
        residual_rows.append({"residual_gain": gain, "nominal": nom_m, "stressed": str_m})
        print(f"{gain:>8.2f}{nom_m['travel_mm']:>11.3f}"
              f"{nom_m['residual_share_extension']*100:>12.1f}%"
              f"{str_m['rel_peak_extension']*100:>12.1f}%{str_m['peak_extension_err_mm']:>9.3f}")
    payload["residual_axis"] = residual_rows

    tap.close()
    print("\ntransmitted = peak |x(stressed) - x(nominal)| / nominal extension range,")
    print("              both rollouts taken at the SAME absorber setting")
    print("residual dx = share of the extension increment from the learned heads")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
