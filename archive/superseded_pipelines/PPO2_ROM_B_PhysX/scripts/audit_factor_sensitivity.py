#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Phase B1: how much work does each ROM-B factor actually do?

Offline, no simulator. Drives the command ROM open-loop from an identical initial
state with identical commands, varying only the hidden factors, and reports the
resulting cylinder-extension divergence.

Method notes that matter:

* The initial state is drawn from a dedicated generator, because `mode="random"`
  consumes the global RNG during factor sampling and would otherwise give the
  compared rollouts different starting extensions - an artifact large enough to
  swamp the effect being measured.
* A single-step query at the model's input mean is NOT representative: the ROM is
  recurrent and factor effects compound. Single-step, `delta_extension` moves by
  ~3 ppm; over 240 steps it moves by millimetres.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PPO2_ROM_B_PhysX.runtime import (  # noqa: E402
    FACTOR_NAMES,
    ROMBCommandCylinderROMRuntime,
    ROMBFactorConfig,
    health_scale_to_factors,
)

FACTORS = FACTOR_NAMES
RANGES = {"e_comp": (-0.125, 0.020)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument(
        "--command_rom_checkpoint",
        default=str(
            ROOT / "ROM-cylinders_pipeline/checkpoints"
            / "plant_reachable_v1_finetune_blend25/best_rom.pt"
        ),
    )
    p.add_argument("--output", default="")
    a = p.parse_args()

    rom = Path(a.command_rom_checkpoint).resolve()
    n, steps = int(a.num_envs), int(a.steps)
    init = torch.Generator().manual_seed(int(a.seed))
    ext0 = torch.rand(n, 4, generator=init) * 0.4 + 0.25

    def rollout(fixed: torch.Tensor | None) -> torch.Tensor:
        cfg = ROMBFactorConfig(
            mode=("nominal" if fixed is None else "random"), randomization_scale=1.0
        )
        rt = ROMBCommandCylinderROMRuntime(
            checkpoint=rom, num_envs=n, device="cpu",
            control_dt_s=1.0 / 60.0, factor_cfg=cfg,
        )
        if fixed is not None:
            rt.true_factors.copy_(
                fixed.reshape(1, len(FACTORS)).expand(n, -1)
                if fixed.ndim == 1 else fixed
            )
        ext = ext0.clone()
        vel = torch.zeros(n, 4)
        dp = torch.zeros(n, 4)
        for k in range(steps):
            cmd = torch.zeros(n, 8)
            cmd[:, :4] = 0.03 * torch.sin(torch.tensor(k / 25.0))
            out = rt.step(
                command=cmd, extension=ext, extension_velocity=vel, dP=dp
            )
            ext = ext + out.delta_extension
            vel = out.extension_velocity_next
            dp = dp + out.delta_dP
        return ext

    base = rollout(None)
    report: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command_rom": str(rom),
        "num_envs": n,
        "steps": steps,
        "method": "open-loop recurrent rollout, identical initial state and commands",
    }

    print("SINGLE-FACTOR SENSITIVITY (all other factors nominal)\n")
    print(f"  {'factor':12s} {'value':>8s} {'mean divergence':>17s} {'p95':>10s}")
    singles = []
    for index, name in enumerate(FACTORS):
        values = RANGES.get(name, (1.0,))
        for value in values:
            fixed = torch.zeros(len(FACTORS))
            fixed[index] = value
            d = (rollout(fixed) - base).abs().mean(dim=-1)
            singles.append({
                "factor": name, "value": float(value),
                "mean_divergence_m": float(d.mean()),
                "p95_divergence_m": float(d.quantile(0.95)),
            })
            print(f"  {name:12s} {value:>8.3f} {float(d.mean()):>17.5f} "
                  f"{float(d.quantile(0.95)):>10.5f}")
    report["single_factor"] = singles

    fixed = torch.zeros(len(FACTORS))
    fixed[1:] = 1.0
    d = (rollout(fixed) - base).abs().mean(dim=-1)
    report["all_seals_max"] = {
        "mean_divergence_m": float(d.mean()),
        "p95_divergence_m": float(d.quantile(0.95)),
    }
    print(f"\n  {'all seals=1.0':21s} {float(d.mean()):>17.5f} "
          f"{float(d.quantile(0.95)):>10.5f}")

    print("\nH CURVE (correlated, evaluation-only coordinate)\n")
    curve = []
    for h in (0.4, 0.6, 0.8, 1.0, 1.2):
        d = (rollout(health_scale_to_factors(h)[0]) - base).abs().mean(dim=-1)
        curve.append({"h": h, "mean_divergence_m": float(d.mean())})
        print(f"  h={h:.1f}  mean divergence = {float(d.mean()):.5f} m")
    report["h_curve"] = curve

    gen = torch.Generator().manual_seed(int(a.seed) + 1)
    u = torch.rand(n, len(FACTORS), generator=gen)
    drawn = torch.stack(
        [-0.125 + u[:, 0] * 0.145, *[u[:, 1 + j] for j in range(4)]], dim=-1,
    )
    d = (rollout(drawn) - base).abs().mean(dim=-1)
    correlations = {
        name: float(torch.corrcoef(torch.stack([drawn[:, i], d]))[0, 1])
        for i, name in enumerate(FACTORS)
    }
    report["independent_sampling"] = {
        "mean_divergence_m": float(d.mean()),
        "p95_divergence_m": float(d.quantile(0.95)),
        "max_divergence_m": float(d.max()),
        "correlation_with_divergence": correlations,
    }
    print("\nINDEPENDENT UNIFORM SAMPLING (the training distribution)\n")
    print(f"  mean={float(d.mean()):.5f}  p95={float(d.quantile(0.95)):.5f}  "
          f"max={float(d.max()):.5f} m")
    print("\n  correlation of divergence with each factor:")
    for name, value in correlations.items():
        flag = "  <-- carries the randomization" if abs(value) > 0.1 else ""
        print(f"    {name:10s} {value:+.3f}{flag}")

    inert = [
        row["factor"] for row in singles
        if row["mean_divergence_m"] < 0.0005
    ]
    report["inert_factors"] = sorted(set(inert))
    if inert:
        print(
            f"\n  INERT (<0.5 mm at their extreme): {sorted(set(inert))}\n"
            "  A factor that cannot move the plant cannot support a robustness\n"
            "  claim. Either make it identifiable in the ROM or remove it from\n"
            "  the declared uncertainty set."
        )

    out = (
        Path(a.output).expanduser().resolve()
        if a.output
        else ROOT / "PPO2_ROM_B_PhysX/reports"
        / f"factor_sensitivity_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
