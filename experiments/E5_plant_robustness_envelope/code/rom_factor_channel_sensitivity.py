#!/usr/bin/env python
"""E5 Stage 1b: the `system_factors` channel ranking, on the corrected harness.

CPU-only, no Isaac. Re-derives the flow / compliance / seal / pressure ranking
that `AAM_module/tools/rom_sensitivity.py` produced -- the ranking that fed the
Stage 4/5 factor selection and the AAM's factor ranges -- using the deployed
bridge and the all-axes probe instead.

Why the old ranking needs rechecking
------------------------------------
`rom_sensitivity.py` has two defects that both suppress and distort sensitivity:

1. **Single-axis probe.** Its trajectory drives one cylinder at a time, and
   `command_rom_direction_guard` scales its limit by |command|, so three of four
   axes sit at zero command with their response forced to exactly zero. On the
   bulk-modulus grid this cost a factor of 122 in apparent degradation.
2. **Free-running rollout.** It feeds the ROM its own `next_extension` and
   starts at -0.5 m, outside every cylinder's limits ([0.018, 0.493] and up).
   The rollout diverges to 291 mm of travel on +-1 mm commands, so the ranking
   is measured far off the training manifold.

This screen keeps the same question and fixes both: deployed runtime, all-axes
probe, measured-velocity feedback, guard on, extension clamped to the
checkpoint's own cylinder limits.

Each channel is swept to its own audited range endpoints (the values the
deployment actually draws from), not a common delta, because that is what the
AAM's ranges are defined over.

Usage:
    python3.11 experiments/E5_plant_robustness_envelope/code/rom_factor_channel_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rom_stress_lib import (  # noqa: E402
    ROOT, TRAJECTORIES, OutputTap, build_runtime, metrics, rollout,
)

# Audited ranges, from verge_task/deployment_factors.py RANGES.
# HydraulicROM's channel order is [compliance, flow, pressure, seal x4].
CHANNELS = {
    "flow":       {"slot": 1, "range": (-0.25, 0.25)},
    "compliance": {"slot": 0, "range": (-0.25, 0.10)},
    "pressure":   {"slot": 2, "range": (-0.15, 0.15)},
    "seal":       {"slot": None, "range": (0.0, 1.0)},   # occupies slots 3..6
}


def factor_vector(channel: str, value: float, device: torch.device) -> torch.Tensor:
    v = torch.zeros(1, 7, device=device)
    spec = CHANNELS[channel]
    if spec["slot"] is None:
        v[0, 3:7] = value
    else:
        v[0, spec["slot"]] = value
    return v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"))
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(ROOT / "experiments/E5_plant_robustness_envelope/results/rom_factor_channel_sensitivity.json"))
    args = p.parse_args()

    torch.manual_seed(17)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    runtime = build_runtime(args.checkpoint, device)
    limits = runtime.checkpoint["cylinder_limits"].to(device)
    tap = OutputTap(runtime.model)

    payload: dict = {"checkpoint": args.checkpoint, "ticks": args.ticks, "probes": {}}

    for probe in ("all_axes", "staggered"):
        command = TRAJECTORIES[probe](args.ticks, device)
        nominal = rollout(runtime, command, limits, guard=True, tap=tap)
        nom_metrics = metrics(nominal, nominal)
        rows = []
        for name, spec in CHANNELS.items():
            lo, hi = spec["range"]
            low = rollout(runtime, command, limits, guard=True, tap=tap,
                          system_factors=factor_vector(name, lo, device))
            high = rollout(runtime, command, limits, guard=True, tap=tap,
                           system_factors=factor_vector(name, hi, device))
            m_lo, m_hi = metrics(nominal, low), metrics(nominal, high)
            span = hi - lo
            # normalized sensitivity across the audited range, per signal
            S = {sig: ((high[sig] - low[sig]).abs().mean() / span).item()
                 for sig in ("dP", "velocity", "extension")}
            rows.append({
                "channel": name, "range": [lo, hi],
                "S_dP": S["dP"], "S_velocity": S["velocity"], "S_extension": S["extension"],
                "rel_extension": max(m_lo["rel_peak_extension"], m_hi["rel_peak_extension"]),
                "rel_dP": max(m_lo["rel_peak_dP"], m_hi["rel_peak_dP"]),
                "err_mm": max(m_lo["peak_extension_err_mm"], m_hi["peak_extension_err_mm"]),
                "guard_veto_rate": m_hi["guard_veto_rate"],
            })
        rows.sort(key=lambda r: r["rel_extension"], reverse=True)
        payload["probes"][probe] = {"nominal": nom_metrics, "ranking": rows}

        print(f"\n=== {probe} | nominal travel {nom_metrics['travel_mm']:.2f} mm, "
              f"guard veto {nom_metrics['guard_veto_rate']*100:.1f}% ===")
        print(f"{'channel':<12}{'audited range':>16}{'rel ext':>10}{'err mm':>10}"
              f"{'rel dP':>10}{'S_ext':>11}")
        for r in rows:
            lo, hi = r["range"]
            print(f"{r['channel']:<12}{f'[{lo}, {hi}]':>16}"
                  f"{r['rel_extension']*100:>9.1f}%{r['err_mm']:>10.3f}"
                  f"{r['rel_dP']*100:>9.1f}%{r['S_extension']:>11.3e}")

    tap.close()
    order = {p: [r["channel"] for r in payload["probes"][p]["ranking"]] for p in payload["probes"]}
    by_S = {p: [r["channel"] for r in
                sorted(payload["probes"][p]["ranking"], key=lambda r: r["S_extension"], reverse=True)]
            for p in payload["probes"]}
    payload["ranking_order"] = {"by_rel_extension": order, "by_S_extension": by_S}

    # rom_sensitivity.py's own published numbers, for the comparison.
    legacy = {
        "by_rel_extension": ["seal", "flow", "compliance", "pressure"],
        "by_S_extension": ["flow", "pressure", "compliance", "seal"],
    }
    print("\nranking over the audited range (rel_extension) -- what factor ranges are chosen on:")
    for probe, names in order.items():
        print(f"  {probe:<11}{' > '.join(names)}")
    print(f"  {'legacy':<11}{' > '.join(legacy['by_rel_extension'])}")
    print("\nranking per unit severity (S_extension) -- divides by range width:")
    for probe, names in by_S.items():
        print(f"  {probe:<11}{' > '.join(names)}")
    print(f"  {'legacy':<11}{' > '.join(legacy['by_S_extension'])}")
    payload["legacy_ranking"] = legacy
    payload["order_unchanged"] = {
        "by_rel_extension": order["all_axes"] == legacy["by_rel_extension"],
        "by_S_extension": by_S["all_axes"] == legacy["by_S_extension"],
    }
    print(f"\norder unchanged by the probe fix: {payload['order_unchanged']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
