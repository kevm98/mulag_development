"""Paired R vs IF statistics over the 33 held-out (mesh, family) cells.

Answers the three things Table 7.1 does not report: a paired bootstrap interval
on the cross-stack difference, the median paired difference, and whether the
aggregate survives dropping any one trajectory family.
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E1 = ROOT / "outputs/e1_tracking"
MESHES = ["1337", "17", "909"]
STACKS = ["ppo_rom", "ppo_id_fd"]
B = 20000


def cells(stack, key):
    out = {}
    for m in MESHES:
        blk = json.load(open(E1 / f"{stack}_s17_m{m}.json"))["per_mesh"][m]
        for fam in blk["path_family_results"]:
            out[(m, fam["family"])] = fam[key]
    return out


def boot_ci(diffs, stat, b=B, seed=0):
    random.seed(seed)
    n = len(diffs)
    draws = sorted(stat([diffs[random.randrange(n)] for _ in range(n)])
                   for _ in range(b))
    return draws[int(0.025 * b)], draws[int(0.975 * b)]


def report(key, unit, scale):
    r = cells("ppo_rom", key)
    f = cells("ppo_id_fd", key)
    keys = sorted(set(r) & set(f))
    fams = sorted({k[1] for k in keys})
    # Positive difference = IF better (lower error).
    d = [(r[k] - f[k]) * scale for k in keys]
    print(f"\n=== {key}  ({unit}) ===")
    print(f"cells={len(d)}  families={len(fams)}")
    print(f"mean paired difference   {st.mean(d):+8.2f}  "
          f"CI {boot_ci(d, st.mean)[0]:+.2f} .. {boot_ci(d, st.mean)[1]:+.2f}")
    print(f"median paired difference {st.median(d):+8.2f}  "
          f"CI {boot_ci(d, st.median)[0]:+.2f} .. {boot_ci(d, st.median)[1]:+.2f}")
    print(f"cells where IF better    {sum(1 for v in d if v > 0)}/{len(d)}")

    fam_mean = {fa: st.mean([(r[k] - f[k]) * scale for k in keys if k[1] == fa])
                for fa in fams}
    print("  per-family mean difference (IF better = positive):")
    for fa in sorted(fam_mean, key=lambda a: fam_mean[a]):
        print(f"    {fa:24s} {fam_mean[fa]:+9.2f}")
    print(f"  families where IF better: "
          f"{sum(1 for v in fam_mean.values() if v > 0)}/{len(fams)}")

    print("  leave-one-family-out mean over remaining cells:")
    flips = []
    full = st.mean(d)
    for fa in fams:
        kept = [(r[k] - f[k]) * scale for k in keys if k[1] != fa]
        m = st.mean(kept)
        lo, hi = boot_ci(kept, st.mean)
        flag = "  SIGN FLIPS" if (m > 0) != (full > 0) else ""
        if flag:
            flips.append(fa)
        print(f"    drop {fa:24s} {m:+9.2f}  CI {lo:+.2f} .. {hi:+.2f}{flag}")
    print(f"  full-set mean {full:+.2f}; sign flips on: {flips or 'none'}")

    # Trimmed aggregate: drop the most extreme family in each direction.
    order = sorted(fams, key=lambda a: fam_mean[a])
    trimmed = [(r[k] - f[k]) * scale for k in keys
               if k[1] not in (order[0], order[-1])]
    lo, hi = boot_ci(trimmed, st.mean)
    print(f"  trimmed (drop {order[0]} and {order[-1]}): "
          f"{st.mean(trimmed):+.2f}  CI {lo:+.2f} .. {hi:+.2f}")


if __name__ == "__main__":
    report("mean_abs_lateral_error_m", "mm", 1000.0)
    report("mean_abs_cut_height_error_m", "mm", 1000.0)
