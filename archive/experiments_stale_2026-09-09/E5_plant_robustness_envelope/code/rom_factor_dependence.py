#!/usr/bin/env python
"""E5 Stage 1: which of the ROM's physical parameters can degrade it at all?

CPU-only, no Isaac. Sweeps every physical parameter in `HydraulicROM` one at a
time over a wide log grid, on the all-axes probe, through the deployed bridge,
and ranks them by how far each can actually move the realised extension.

Stage 0b established that effective bulk modulus is a live degradation axis.
This asks the prior question: of the eight physical parameters plus the two
structural offsets, which are usable stress axes and which are inert -- and for
the inert ones, whether that is a property of the model or an artefact of the
clamp.

Two properties this screen exists to measure
--------------------------------------------
1. **Rail-bound parameters.** Training drove five of the eight physical
   parameters to the clamp bounds `HydraulicROM.forward` applies:
   flow_gain and effective_area sit on their floors, effective_mass and
   viscous_friction on their ceilings, and supply_pressure is split with two
   cylinders on each rail. Perturbing *toward* a rail delivers nothing. So
   every row reports the delivered scale after clamping, computed against the
   clamped nominal -- never the requested one.
2. **Reach, separately per direction.** A parameter can be a good stress axis
   downward and inert upward. One symmetric number would hide that, so
   down-reach and up-reach are reported apart.

Usage:
    python3.11 experiments/E5_plant_robustness_envelope/code/rom_factor_dependence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rom_stress_lib import (  # noqa: E402
    ROOT, PARAM_CLAMPS, TRAJECTORIES, OutputTap, build_runtime,
    delivered, get_param, metrics, rollout, set_param, set_raw,
)

# Wide and log-spaced: several parameters need orders of magnitude before they
# leave their rail, and a narrow grid would report those as inert.
SCALES = (0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0)
# deadzone width is 0.25*sigmoid(raw_width), so shifts act on the logit.
RAW_WIDTH_SHIFTS = (-3.0, -1.5, 0.0, 1.5, 3.0)
# extension_reference sets the zero of the volume modulation term, in metres.
EXTENSION_REF_SHIFTS = (-0.05, -0.02, 0.0, 0.02, 0.05)


def multiplicative_factors() -> tuple[str, ...]:
    return tuple(PARAM_CLAMPS)


def sweep_multiplicative(runtime, key, command, limits, nominal_run, *, guard, tap):
    """Scale one clamped physical parameter over the log grid."""
    model = runtime.model
    nominal = get_param(model, key)
    # The nominal itself may be clamped, so the reference is the DELIVERED
    # nominal. Otherwise a rail-pinned parameter reports a scale it never had.
    nominal_delivered = delivered(key, nominal)
    rows = []
    for scale in SCALES:
        requested = nominal * scale
        got = delivered(key, requested)
        # geometric mean over cylinders: the grid is multiplicative
        delivered_scale = (got / nominal_delivered).log().mean().exp().item()
        set_param(model, key, requested)
        run = rollout(runtime, command, limits, guard=guard, tap=tap)
        rows.append({
            "scale_requested": scale,
            "scale_delivered": delivered_scale,
            "inert": abs(delivered_scale - 1.0) < 1.0e-6 and abs(scale - 1.0) > 1.0e-6,
            "value_delivered": got.tolist(),
            **metrics(nominal_run, run),
        })
    set_param(model, key, nominal)
    return rows


def sweep_additive(runtime, key, shifts, command, limits, nominal_run, *, guard, tap):
    """Shift a parameter that is not log-parameterised."""
    model = runtime.model
    nominal = model.state_dict()[key].clone()
    rows = []
    for shift in shifts:
        set_raw(model, key, nominal + shift)
        run = rollout(runtime, command, limits, guard=guard, tap=tap)
        rows.append({
            "shift": shift,
            "scale_requested": shift,
            "scale_delivered": shift,
            "inert": False,
            "value_delivered": (nominal + shift).tolist(),
            **metrics(nominal_run, run),
        })
    set_raw(model, key, nominal)
    return rows


def summarise(name: str, rows: list[dict]) -> dict:
    """Reachable degradation, and how much of the grid the clamp ate."""
    live = [r for r in rows if not r["inert"]]
    moved = [r for r in live if abs(r["scale_delivered"] - 1.0) > 1.0e-6]
    worst = max(rows, key=lambda r: r["rel_peak_extension"])
    down = [r["scale_delivered"] for r in moved if r["scale_delivered"] < 1.0]
    up = [r["scale_delivered"] for r in moved if r["scale_delivered"] > 1.0]
    return {
        "factor": name,
        "max_rel_extension": worst["rel_peak_extension"],
        "max_extension_err_mm": worst["peak_extension_err_mm"],
        "at_scale_requested": worst["scale_requested"],
        "at_scale_delivered": worst["scale_delivered"],
        "max_rel_dP": max(r["rel_peak_dP"] for r in rows),
        "down_reach": min(down) if down else 1.0,
        "up_reach": max(up) if up else 1.0,
        "inert_points": sum(1 for r in rows if r["inert"]),
        "grid_points": len(rows),
        "residual_share_extension": worst.get("residual_share_extension", float("nan")),
        "all_finite": all(r["finite"] for r in rows),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"))
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--device", default="cpu")
    p.add_argument("--trajectory", choices=tuple(TRAJECTORIES), default="all_axes")
    p.add_argument("--out", default=str(ROOT / "experiments/E5_plant_robustness_envelope/results/rom_factor_dependence.json"))
    args = p.parse_args()

    torch.manual_seed(17)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    runtime = build_runtime(args.checkpoint, device)
    model = runtime.model
    limits = runtime.checkpoint["cylinder_limits"].to(device)
    command = TRAJECTORIES[args.trajectory](args.ticks, device)
    tap = OutputTap(model)

    print(f"checkpoint : {args.checkpoint}")
    print(f"probe      : {args.trajectory}   ticks: {args.ticks}   device: {device}")
    print("\nlearned parameters against the clamps `forward` applies:")
    print(f"{'parameter':<22}{'learned (min..max)':>26}{'clamp':>24}{'  rail'}")
    for key, (lo, hi) in PARAM_CLAMPS.items():
        v = get_param(model, key)
        rail = []
        if (v.min() / lo).item() < 1.01:
            rail.append("FLOOR")
        if (hi / v.max()).item() < 1.01:
            rail.append("CEIL")
        print(f"{key:<22}{f'{v.min():.3e}..{v.max():.3e}':>26}"
              f"{f'[{lo:.0e}, {hi:.0e}]':>24}  {'+'.join(rail) or '-'}")

    payload: dict = {
        "checkpoint": args.checkpoint,
        "ticks": args.ticks,
        "trajectory": args.trajectory,
        "scales": list(SCALES),
        "guard_modes": {},
    }

    for guard in (True, False):
        guard_key = "guard_on" if guard else "guard_off"
        nominal_run = rollout(runtime, command, limits, guard=guard, tap=tap)
        nominal_metrics = metrics(nominal_run, nominal_run)
        factors: dict[str, list[dict]] = {}
        for key in multiplicative_factors():
            factors[key] = sweep_multiplicative(
                runtime, key, command, limits, nominal_run, guard=guard, tap=tap
            )
        factors["deadzone.raw_width"] = sweep_additive(
            runtime, "deadzone.raw_width", RAW_WIDTH_SHIFTS,
            command, limits, nominal_run, guard=guard, tap=tap,
        )
        factors["extension_reference"] = sweep_additive(
            runtime, "extension_reference", EXTENSION_REF_SHIFTS,
            command, limits, nominal_run, guard=guard, tap=tap,
        )
        summary = sorted(
            (summarise(name, rows) for name, rows in factors.items()),
            key=lambda s: s["max_rel_extension"], reverse=True,
        )
        payload["guard_modes"][guard_key] = {
            "nominal": nominal_metrics, "factors": factors, "ranking": summary,
        }

        print(f"\n=== {guard_key} | nominal travel {nominal_metrics['travel_mm']:.2f} mm, "
              f"guard veto {nominal_metrics['guard_veto_rate']*100:.1f}%, "
              f"residual share of dx {nominal_metrics['residual_share_extension']*100:.1f}% ===")
        print(f"{'factor':<22}{'max rel ext':>12}{'err mm':>10}{'at scale':>10}"
              f"{'delivered':>11}{'down':>9}{'up':>9}{'inert':>8}{'max rel dP':>12}")
        for s in summary:
            inert = f"{s['inert_points']}/{s['grid_points']}"
            print(f"{s['factor']:<22}{s['max_rel_extension']*100:>11.1f}%{s['max_extension_err_mm']:>10.2f}"
                  f"{s['at_scale_requested']:>10g}{s['at_scale_delivered']:>11.3g}"
                  f"{s['down_reach']:>9.3g}{s['up_reach']:>9.3g}{inert:>8}"
                  f"{s['max_rel_dP']*100:>11.1f}%")

    tap.close()
    print("\nmax rel ext = largest peak |x(perturbed) - x(nominal)| / nominal extension range")
    print("delivered   = post-clamp scale actually applied, against the CLAMPED nominal")
    print("down/up     = furthest scale reached in each direction; 1.0 means the clamp blocked it")
    print("inert       = grid points where the clamp delivered no change at all")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
