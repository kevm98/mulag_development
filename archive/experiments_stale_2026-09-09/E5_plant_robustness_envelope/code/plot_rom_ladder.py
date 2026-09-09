#!/usr/bin/env python3
"""Plot the ROM actuator-authority robustness curve.

Ground contact against authority alpha, with the across-mesh range as the error
band and the predeclared thresholds drawn as fixed horizontal lines. The
thresholds are constants from the protocol, not fits, and are drawn even when
no level crosses them -- a threshold visible and uncrossed is a result.

    python plot_rom_ladder.py --summary experiments/E5_plant_robustness_envelope/results/rom_authority_ladder.json \
                              --out experiments/E5_plant_robustness_envelope/figures/ch7_e5_rom_authority.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ONSET_DROP_PP = 5.0
BREAKDOWN_CONTACT = 0.700


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-meshes", type=int, default=3)
    p.add_argument("--logx", action="store_true",
                   help="log alpha axis; the whole knee lives in the last 5%% of "
                        "a linear axis and is unreadable there")
    a = p.parse_args()

    s = json.loads(Path(a.summary).read_text())
    levels = [r for r in s["levels"] if r["n_meshes"] >= a.min_meshes]
    thin = [r for r in s["levels"] if r["n_meshes"] < a.min_meshes]
    levels.sort(key=lambda r: r["alpha"])
    thin.sort(key=lambda r: r["alpha"])

    nom = s["nominal"]["contact"]
    onset_line = nom - ONSET_DROP_PP / 100.0

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    if levels:
        x = [r["alpha"] for r in levels]
        y = [r["contact"]["mean"] for r in levels]
        lo = [r["contact"]["min"] for r in levels]
        hi = [r["contact"]["max"] for r in levels]
        ax.fill_between(x, lo, hi, color="#D31F26", alpha=0.18,
                        label="across-mesh range")
        ax.plot(x, y, "-o", color="#D31F26", lw=2, ms=5,
                label=f"worst-family ground contact (n={a.min_meshes} meshes)")
        wm = [r.get("worst_mesh_contact") for r in levels]
        if all(v is not None for v in wm):
            ax.plot(x, wm, "--s", color="#7A1216", lw=1.4, ms=4, alpha=0.9,
                    label="worst mesh (the gate metric)")

    # Levels measured on fewer meshes are shown, but visibly distinguished:
    # they carry no error estimate and must not be read as equally supported.
    if thin:
        ax.plot([r["alpha"] for r in thin], [r["contact"]["mean"] for r in thin],
                "x", color="#888888", ms=6, ls="none",
                label="single mesh (no error estimate)")

    ax.axhline(nom, color="#333333", lw=1, ls=":", label=f"nominal = {nom:.3f}")
    ax.axhline(onset_line, color="#C8860D", lw=1.4, ls="--",
               label=f"degradation onset ({ONSET_DROP_PP:.0f} pp below nominal)")
    ax.axhline(BREAKDOWN_CONTACT, color="#0F8A3C", lw=1.4, ls="-.",
               label=f"functional breakdown ({BREAKDOWN_CONTACT})")

    onset_a = s.get("degradation_onset_alpha")
    brk_a = s.get("functional_breakdown_alpha")
    if onset_a is not None:
        ax.axvline(onset_a, color="#C8860D", lw=0.9, alpha=0.5)
    if brk_a is not None:
        ax.axvline(brk_a, color="#0F8A3C", lw=0.9, alpha=0.5)

    ax.set_xlabel(r"actuator authority $\alpha$  (fraction of realised cylinder motion)")
    ax.set_ylabel("ground contact fraction")
    ax.set_title("PPO-ROM: plant robustness envelope under actuator-authority loss")
    if a.logx:
        ax.set_xscale("symlog", linthresh=0.005)
        ax.set_xlim(0, 1.1)
    else:
        ax.set_xlim(-0.03, 1.03)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)

    plateau = s.get("plateau_min_alpha")
    caption = []
    if plateau is not None:
        caption.append(f"robust to alpha >= {plateau:g}")
    caption.append(f"onset at alpha = {onset_a}" if onset_a is not None
                   else "onset not resolved")
    caption.append(f"breakdown at alpha = {brk_a}" if brk_a is not None
                   else f"breakdown (<{BREAKDOWN_CONTACT}) not reached")
    fig.text(0.01, 0.005, "   |   ".join(caption), fontsize=7.5, color="#444444")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(a.out, dpi=200)
    png = str(Path(a.out).with_suffix(".png"))
    fig.savefig(png, dpi=200)
    print(f"wrote {a.out}\nwrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
