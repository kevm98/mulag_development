#!/usr/bin/env python3
"""Two-panel ROM authority robustness figure.

Panel A -- the robustness curve, on band-mean ground contact. This is the
aggregate over distance bands and meshes. It is monotone in alpha and is the
curve to read for the degradation envelope.

Panel B -- the project's own acceptance gate,
``worst_mesh_worst_family_ground_contact_fraction`` against its 0.700
threshold, with the safety-termination rate on a twin axis.

Both panels are shown because neither alone is honest.

The gate metric is NOT monotone: contact is lower at alpha=1.00 (0.741) than
at alpha=0.20 (0.951). That is not noise and not a broken harness. At full
authority the policy is aggressive enough to trigger safety terminations
(0.0487, the highest of any level, falling to ~0.001 as authority is removed),
and a terminated episode scores as lost contact. So the gate metric mixes two
distinct failure modes -- contact lost to over-aggression at high alpha, and
contact lost to actuator starvation at low alpha -- and only the second is the
robustness question this experiment asks.

Reporting only panel A would hide that the deployed policy passes its own gate
at nominal by just 4.2 pp. Reporting only panel B would present a
non-monotone curve as a degradation envelope. Hence both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BREAKDOWN_CONTACT = 0.700
ONSET_DROP_PP = 5.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    s = json.loads(Path(a.summary).read_text())
    lv = sorted([r for r in s["levels"] if r["n_meshes"] >= 3], key=lambda r: r["alpha"])
    x = [r["alpha"] for r in lv]
    band = [r["band_mean_contact"]["mean"] for r in lv]
    band_lo = [r["band_mean_contact"]["min"] for r in lv]
    band_hi = [r["band_mean_contact"]["max"] for r in lv]
    gate = [r["worst_mesh_contact"] for r in lv]
    saf = [r["safety"]["mean"] for r in lv]

    nom_band = [r for r in lv if r["alpha"] == 1.0][0]["band_mean_contact"]["mean"]
    onset_line = nom_band - ONSET_DROP_PP / 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # ---- Panel A: the robustness curve -----------------------------------
    ax1.fill_between(x, band_lo, band_hi, color="#D31F26", alpha=0.18,
                     label="across-mesh range")
    ax1.plot(x, band, "-o", color="#D31F26", lw=2, ms=5, label="ground contact (band mean)")
    ax1.axhline(nom_band, color="#333333", lw=1, ls=":", label=f"nominal = {nom_band:.3f}")
    ax1.axhline(onset_line, color="#C8860D", lw=1.4, ls="--",
                label=f"onset ({ONSET_DROP_PP:.0f} pp below nominal)")
    ax1.axhline(BREAKDOWN_CONTACT, color="#0F8A3C", lw=1.4, ls="-.",
                label=f"breakdown ({BREAKDOWN_CONTACT})")
    ax1.set_xscale("symlog", linthresh=0.005)
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel(r"actuator authority $\alpha$")
    ax1.set_ylabel("ground contact fraction")
    ax1.set_title("A  robustness envelope (band mean)")
    ax1.grid(alpha=0.25, lw=0.6)
    ax1.legend(fontsize=7.5, loc="lower right", framealpha=0.95)

    # ---- Panel B: the acceptance gate, and the confound -------------------
    ax2.plot(x, gate, "--s", color="#7A1216", lw=1.8, ms=5,
             label="worst-mesh worst-family (gate metric)")
    ax2.axhline(BREAKDOWN_CONTACT, color="#0F8A3C", lw=1.4, ls="-.",
                label=f"gate threshold ({BREAKDOWN_CONTACT})")
    ax2.set_xscale("symlog", linthresh=0.005)
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel(r"actuator authority $\alpha$")
    ax2.set_ylabel("worst-mesh worst-family contact")
    ax2.set_title("B  acceptance gate, and the aggression confound")
    ax2.grid(alpha=0.25, lw=0.6)

    ax2b = ax2.twinx()
    ax2b.plot(x, saf, ":^", color="#1F4E9C", lw=1.4, ms=4,
              label="safety termination rate")
    ax2b.set_ylabel("safety termination rate", color="#1F4E9C")
    ax2b.tick_params(axis="y", colors="#1F4E9C")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right", framealpha=0.95)

    ax2.annotate("nominal is an outlier:\naggressive policy ->\nsafety terminations",
                 xy=(1.0, gate[-1]), xytext=(0.02, 0.90),
                 fontsize=7.5, color="#444444",
                 arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))

    fig.suptitle("PPO-ROM plant robustness under actuator-authority loss "
                 "(3 terrain meshes, 11 path families, 4 km/h)", fontsize=11)
    fig.text(0.005, 0.005,
             f"plateau: contact flat for alpha >= 0.05   |   "
             f"onset (band mean): alpha = 0.03   |   "
             f"gate breakdown: alpha = 0.01   |   "
             f"nominal passes its own gate by 4.2 pp",
             fontsize=7.5, color="#444444")

    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=200)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
