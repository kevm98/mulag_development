"""The task-space oracle, re-read under the corrected layout.

The original sweep put every offset in ONE process and failed its A/A: a repeat
of the baseline scored -61.8% cut and -43.1% lateral against itself
(RESULT_AA_sweep_invalid_2026-09-06.md). Its negative was withdrawn, so this is
not a confirmation run. The question is open.

Here each offset is its own process, paired by environment index under common
random numbers, which is invariant 3's other layout and the one whose A/A
passes.

Statistic and bounds are triplet.py's. The clearance guard is kept, because
cut_height_error is relu(h - reference) and zero in contact, so an offset that
drives the cutter into the ground scores a perfect cut height by destroying
clearance.

Filenames encode the offset with '.' as 'p' and '-' as 'm', e.g. ym0p02_z0 is
b_Y = -0.02, b_Z = 0.

Usage:  phi_redo_table.py [--dir outputs/phi_redo] [--mesh 2718]
"""
import glob
import json
import math
import os
import random
import statistics as st
import sys

B = 20000
BOUND = {"cut": 10.0, "lateral": 1.5}
CONTACT_FLOOR = 0.700


def decode(tag):
    y, z = tag.split("_z")
    f = lambda s: float(s.replace("p", ".").replace("m", "-"))
    return f(y[1:]), f(z)


def compare(ref, arm):
    random.seed(0)
    n = min(len(ref), len(arm))
    ref, arm = ref[:n], arm[:n]
    diffs = [x - y for x, y in zip(ref, arm)]
    boot = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)])
                  for _ in range(B))
    lo, hi = boot[int(0.025 * B)], boot[int(0.975 * B)]
    lr = [math.log(x / y) for x, y in zip(ref, arm) if x > 0 and y > 0]
    pct = (math.exp(st.median(lr)) - 1) * 100 if lr else float("nan")
    return dict(pct=pct, sig=(lo > 0 or hi < 0))


def label(res):
    wins = [k for k, r in res.items() if r["pct"] > 0 and r["sig"]]
    if not wins:
        if all(r["pct"] < 0 and r["sig"] for r in res.values()):
            return "both worse", False
        return "no improvement", False
    if len(wins) == 2:
        return "joint improvement", True
    other = "lateral" if wins[0] == "cut" else "cut"
    if res[other]["pct"] >= -BOUND[other]:
        return f"Pareto ({other} maintained)", True
    return f"trade-off ({other} worse)", False


if __name__ == "__main__":
    root = (sys.argv[sys.argv.index("--dir") + 1]
            if "--dir" in sys.argv else "outputs/phi_redo")
    mesh = (sys.argv[sys.argv.index("--mesh") + 1]
            if "--mesh" in sys.argv else "2718")
    rows = {}
    for f in glob.glob(os.path.join(root, f"*_p*_m{mesh}.json")):
        tag = os.path.basename(f).split("_p")[0]
        for r in json.load(open(f)).get("per_mesh", {}).values():
            rows[decode(tag)] = r
            break
    if (0.0, 0.0) not in rows:
        raise SystemExit(f"baseline 0,0 not scored yet ({len(rows)} arms present)")
    base = rows[(0.0, 0.0)]
    b_cut = [v * 1000.0 for v in base["per_env_cut_error_m"]]
    b_lat = [v * 1000.0 for v in base["per_env_lateral_m"]]

    print(f"Task-space oracle, corrected layout (one arm per process), mesh {mesh}.")
    print(f"Paired per-environment differences, B={B}. * = CI excludes zero.")
    print(f"baseline: cut {st.median(b_cut):.2f} mm  lateral {st.median(b_lat):.2f} mm  "
          f"contact {st.mean(base['per_env_contact_fraction']):.3f}  "
          f"safety {base['safety_violation_rate']:.3f}   "
          f"({len(rows) - 1} offsets scored)")
    print(f"\n{'b_Y':>7} {'b_Z':>7}  {'cut':>10} {'d_mm':>8}  {'lateral':>10} "
          f"{'d_mm':>8}  {'contact':>8} {'safety':>7}   outcome")

    winners = []
    for (by, bz), arm in sorted(rows.items()):
        if (by, bz) == (0.0, 0.0):
            continue
        a_cut = [v * 1000.0 for v in arm["per_env_cut_error_m"]]
        a_lat = [v * 1000.0 for v in arm["per_env_lateral_m"]]
        res = {"cut": compare(b_cut, a_cut), "lateral": compare(b_lat, a_lat)}
        lab, ok = label(res)
        contact = arm["worst_family_ground_contact_fraction"]
        if ok and contact < CONTACT_FLOOR:
            lab = f"ARTEFACT: {lab}, contact {contact:.3f}"
            ok = False
        elif ok:
            winners.append(((by, bz), res))
        cell = lambda r: f"{r['pct']:+9.1f}%{'*' if r['sig'] else ' '}"
        print(f"{by:>7.3f} {bz:>7.3f}  {cell(res['cut'])}"
              f"{st.median(a_cut) - st.median(b_cut):+8.2f}  {cell(res['lateral'])}"
              f"{st.median(a_lat) - st.median(b_lat):+8.2f}  "
              f"{st.mean(arm['per_env_contact_fraction']):8.3f} "
              f"{arm['safety_violation_rate']:7.3f}   {lab}")

    if winners:
        best = max(winners, key=lambda w: min(w[1]["cut"]["pct"],
                                              w[1]["lateral"]["pct"]))
        print(f"\n  PASS: best phi = b_Y {best[0][0]}, b_Z {best[0][1]} "
              f"(cut {best[1]['cut']['pct']:+.1f}%, "
              f"lateral {best[1]['lateral']['pct']:+.1f}%)")
    else:
        print(f"\n  No offset passes among the {len(rows) - 1} scored so far.")
