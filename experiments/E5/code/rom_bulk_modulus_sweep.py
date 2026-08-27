#!/usr/bin/env python
"""E5 Stage-0b: is effective bulk modulus a usable ROM degradation axis?

CPU-only, no Isaac. Drives the deployed command ROM through its real deployment
bridge (`CommandCylinderROMRuntime`, the same class `mulag_upper_level_env` uses
in its `command_rom` branch) on one deterministic command trajectory, while the
effective bulk modulus beta is moved off nominal.  Reports how far the ROM's
dP / velocity / realised extension move, and whether the requested stress
actually arrived.

Why this is not the `system_factors` compliance channel
-------------------------------------------------------
`HydraulicROM.forward` pins the compliance factor e_comp to [-0.25, 0.10] and
applies beta <- beta * exp(e_comp).  That path can only deliver 0.779x..1.105x
of nominal beta -- it cannot express the 0x / 0.5x / 2.0x grid at all.  E5's
withdrawn sweep is exactly the failure this causes.  So this sweep writes
`log_bulk_modulus` directly and then reports the *delivered* post-clamp beta,
so a flat curve can never be confused with an undelivered one.

Two grids:
  * scale    -- multiples of the checkpoint's own learned per-cylinder beta
                (E5's 0/0.5/1/1.5/2 convention, nominal = 1.0)
  * absolute -- beta in GPa, the same numbers read as physical stiffness

Plant surrogate
---------------
No PhysX here, so the plant is a perfect follower: the measured extension
integrates the ROM's realised delta_extension and is clamped to the checkpoint's
cylinder limits.  Crucially the measured *velocity* is differentiated from that
measured extension, not carried forward from the ROM's own integrator -- the
deployed env passes `reference.xdot_current` (a plant measurement), and feeding
the ROM its own unanchored velocity instead lets the rollout run away.

Two guard modes, reported separately:
  * guard on  -- `command_rom_direction_guard=True`, what the deployment task
                 configs actually set (`mulag_plant/policy/env_cfg.py`,
                 `verge_task/env_cfg.py`).  Clamps realised motion to
                 3x|command| and zeros any wrong-direction response.
  * guard off -- the raw ROM response.
This separates "beta does not move the ROM" from "beta moves the ROM but the
guard removes it before the plant sees it", which is the distinction E5 says a
null result must make.

Usage:
    python3.11 experiments/E5/code/rom_bulk_modulus_sweep.py
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
    delivered, get_param, metrics, rollout, set_param,
)

BETA_FLOOR_PA, BETA_CEIL_PA = PARAM_CLAMPS["log_bulk_modulus"]
DEFAULT_SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)
DEFAULT_ABSOLUTE_GPA = (0.0, 0.5, 1.0, 1.5, 2.0)


def set_beta(model, beta_pa: torch.Tensor) -> None:
    """Write the requested per-cylinder beta into the model (pre-clamp)."""
    set_param(model, "log_bulk_modulus", beta_pa)


def delivered_beta(beta_pa: torch.Tensor) -> torch.Tensor:
    """What `forward` actually uses after its own clamp."""
    return delivered("log_bulk_modulus", beta_pa)


def report(title: str, axis_name: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(
        f"{axis_name:>8}{'beta_del':>10}{'deliv':>7}"
        f"{'rel dP':>9}{'rel vel':>9}{'rel ext':>9}"
        f"{'travel mm':>11}{'ext err mm':>12}{'auth':>8}{'veto':>8}{'res dx':>9}"
    )
    for r in rows:
        print(
            f"{r['axis']:>8}{r['beta_delivered_gpa']:>10.4f}{r['delivered_fraction']:>7.3f}"
            f"{r['rel_peak_dP']*100:>8.1f}%{r['rel_peak_velocity']*100:>8.1f}%"
            f"{r['rel_peak_extension']*100:>8.1f}%"
            f"{r['travel_mm']:>11.3f}{r['peak_extension_err_mm']:>12.4f}"
            f"{r['authority_kept']*100:>7.1f}%{r['guard_veto_rate']*100:>7.1f}%"
            f"{r.get('residual_share_extension', float('nan'))*100:>8.1f}%"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"))
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--trajectory", choices=(*TRAJECTORIES, "both"), default="both")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(ROOT / "experiments/E5/results/rom_bulk_modulus_sweep.json"))
    args = p.parse_args()

    torch.manual_seed(17)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    runtime = build_runtime(args.checkpoint, device)
    model = runtime.model
    limits = runtime.checkpoint["cylinder_limits"].to(device)
    selected = tuple(TRAJECTORIES) if args.trajectory == "both" else (args.trajectory,)

    tap = OutputTap(model)
    beta_nom = get_param(model, "log_bulk_modulus")
    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {device}   ticks: {args.ticks}   dt: {runtime.sample_time_s} s")
    print(f"nominal beta per cylinder (GPa): {[round(v, 4) for v in (beta_nom / 1e9).tolist()]}")
    print(
        "compliance-factor reach: "
        f"{float(torch.exp(torch.tensor(-0.25))):.3f}x .. {float(torch.exp(torch.tensor(0.10))):.3f}x "
        "of nominal -- the requested grid is unreachable through system_factors, "
        "so log_bulk_modulus is written directly."
    )

    payload: dict = {
        "checkpoint": args.checkpoint,
        "ticks": args.ticks,
        "device": str(device),
        "beta_nominal_gpa": (beta_nom / 1e9).tolist(),
        "beta_clamp_gpa": [BETA_FLOOR_PA / 1e9, BETA_CEIL_PA / 1e9],
        "cylinder_limits_m": limits.tolist(),
        "trajectories": {
            "all_axes": "phase-shifted +-1 mm sinusoids, all four cylinders every tick",
            "staggered": "+-1 mm trapezoid, one cylinder at a time (guard-suppressed probe)",
        },
        "plant_surrogate": "perfect follower; measured velocity differentiated from measured extension",
        "results": {},
    }

    for traj_name in selected:
      command = TRAJECTORIES[traj_name](args.ticks, device)
      payload["results"][traj_name] = {}
      for guard in (True, False):
          guard_key = "guard_on" if guard else "guard_off"
          set_beta(model, beta_nom)
          nominal_run = rollout(runtime, command, limits, guard=guard, tap=tap)
          grids = {}
          for grid_name, axis_values, beta_of in (
              ("scale", DEFAULT_SCALES, lambda s: beta_nom * s),
              ("absolute_gpa", DEFAULT_ABSOLUTE_GPA, lambda g: torch.full_like(beta_nom, g * 1e9)),
          ):
              rows = []
              for value in axis_values:
                  beta_req = beta_of(value)
                  set_beta(model, beta_req)
                  run = rollout(runtime, command, limits, guard=guard, tap=tap)
                  beta_del = delivered_beta(beta_req)
                  rows.append({
                      "axis": f"{value:g}",
                      "beta_requested_gpa": (beta_req.mean() / 1e9).item(),
                      "beta_delivered_gpa": (beta_del.mean() / 1e9).item(),
                      "delivered_fraction": (beta_del.mean() / beta_req.mean().clamp_min(1e-12)).clamp(max=1.0).item(),
                      "beta_delivered_per_cylinder_gpa": (beta_del / 1e9).tolist(),
                      **metrics(nominal_run, run),
                  })
              grids[grid_name] = rows
              label = "multiples of the learned beta" if grid_name == "scale" else "absolute beta in GPa"
              report(
                  f"[{traj_name} | {guard_key}] {label} (E5 0/0.5/1/1.5/2 grid)",
                  "scale" if grid_name == "scale" else "GPa",
                  rows,
              )
          nom_metrics = metrics(nominal_run, nominal_run)
          grids["nominal"] = nom_metrics
          payload["results"][traj_name][guard_key] = grids
          print(
              f"  nominal @ {guard_key}: travel {nom_metrics['travel_mm']:.3f} mm, "
              f"authority kept {nom_metrics['authority_kept']*100:.1f}%, "
              f"guard veto {nom_metrics['guard_veto_rate']*100:.1f}%, "
              f"limit clip {nom_metrics['limit_clip_rate']*100:.1f}%, "
              f"dP/limit {nom_metrics['dP_saturation']*100:.1f}%, "
              f"residual share dP {nom_metrics['residual_share_dP']*100:.1f}% / "
              f"ext {nom_metrics['residual_share_extension']*100:.1f}%"
          )

    print("\nrel  = peak |y(beta) - y(nominal)| / nominal signal range")
    print("deliv= delivered beta / requested beta after the model's [0.01, 3.0] GPa clamp")
    print("auth = sum|delta applied| / sum|delta raw|; veto = guard zeroed the response")
    print("res dx = share of the extension increment coming from the learned residual, not physics")

    tap.close()
    set_beta(model, beta_nom)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
