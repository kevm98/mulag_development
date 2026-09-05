"""The smoothing-oracle table.

Same statistic and success rule as oracle_phi.py, which are triplet.py's and
therefore every previous campaign's: paired per-environment differences against
the baseline arm measured in the SAME process, median, percentile bootstrap
B = 20000, percentage as the median log-ratio, non-inferiority bounds cut 10%
and lateral 1.5%.

Adds the declared secondary outcomes of the root-cause report, because on this
plant they are the mechanism rather than side statistics: ground contact, hard
impacts, p95 tool acceleration and the safety-violation rate.

Usage:  smoothing_table.py [--dir outputs/smoothing] [--mesh 2718]
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
    return dict(pct=pct, sig=(lo > 0 or hi < 0), med_ref=st.median(ref),
                med_arm=st.median(arm))


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


def table(path, payload):
    doc = json.load(open(path))
    block = None
    for mesh in doc.get("per_mesh", {}).values():
        if "paired_smoothing" in mesh:
            block = mesh["paired_smoothing"]
    if block is None:
        print(f"  {payload}: no paired_smoothing block in {path}")
        return None
    arms, key0 = block["arms"], block["baseline_key"]
    base = arms[key0]
    b_cut = [v * 1000.0 for v in base["per_env_cut_error_m"]]
    b_lat = [v * 1000.0 for v in base["per_env_lateral_m"]]

    print(f"\n{'=' * 104}")
    print(f"payload {payload}x   baseline beta:rate = {key0}   "
          f"cut {st.median(b_cut):.2f} mm  lateral {st.median(b_lat):.2f} mm  "
          f"contact {st.mean(base['per_env_contact_fraction']):.3f}  "
          f"impacts {base['hard_impact_step_fraction']:.3f}  "
          f"accel {base['p95_tool_acceleration_mps2']:.1f}  "
          f"safety {base['safety_violation_rate']:.3f}")
    print('=' * 104)
    print(f"{'beta:rate':>12}  {'cut':>10}  {'lateral':>10}  {'contact':>8} "
          f"{'impact':>7} {'accel':>7} {'safety':>7}   outcome")

    winners = []
    for key, arm in arms.items():
        if key == key0:
            continue
        res = {
            "cut": compare(b_cut, [v * 1000.0 for v in arm["per_env_cut_error_m"]]),
            "lateral": compare(b_lat, [v * 1000.0 for v in arm["per_env_lateral_m"]]),
        }
        lab, ok = label(res)
        contact = arm["worst_family_ground_contact_fraction"]
        if ok and contact < CONTACT_FLOOR:
            lab = f"ARTEFACT: {lab}, contact {contact:.3f} < {CONTACT_FLOOR}"
            ok = False
        elif ok:
            winners.append((key, res))
        cell = lambda r: f"{r['pct']:+9.1f}%{'*' if r['sig'] else ' '}"
        print(f"{key:>12}  {cell(res['cut'])}  {cell(res['lateral'])}  "
              f"{st.mean(arm['per_env_contact_fraction']):8.3f} "
              f"{arm['hard_impact_step_fraction']:7.3f} "
              f"{arm['p95_tool_acceleration_mps2']:7.1f} "
              f"{arm['safety_violation_rate']:7.3f}   {lab}")

    if winners:
        best = max(winners, key=lambda w: min(w[1]["cut"]["pct"],
                                              w[1]["lateral"]["pct"]))
        print(f"\n  PASS at {payload}x: best = {best[0]} "
              f"(cut {best[1]['cut']['pct']:+.1f}%, "
              f"lateral {best[1]['lateral']['pct']:+.1f}%)")
        return best[0]
    print(f"\n  FAIL at {payload}x: no smoothing setting meets the rule.")
    return None


if __name__ == "__main__":
    root = (sys.argv[sys.argv.index("--dir") + 1]
            if "--dir" in sys.argv else "outputs/smoothing")
    mesh = (sys.argv[sys.argv.index("--mesh") + 1]
            if "--mesh" in sys.argv else "2718")
    files = sorted(glob.glob(os.path.join(root, f"smooth_m{mesh}_p*.json")),
                   key=lambda f: float(f.rsplit("_p", 1)[1][:-5]))
    if not files:
        raise SystemExit(f"no smooth_m{mesh}_p*.json under {root}")
    print(f"Smoothing oracle, mesh {mesh}. Paired per-environment differences "
          f"against the baseline\narm in the same process. B={B}. "
          f"* = CI excludes zero. Cut bound {BOUND['cut']}%, "
          f"lateral {BOUND['lateral']}%.")
    verdicts = {}
    for path in files:
        payload = path.rsplit("_p", 1)[1][:-5]
        verdicts[payload] = table(path, payload)
    print(f"\n{'=' * 104}\nVERDICT\n{'=' * 104}")
    for payload, best in verdicts.items():
        print(f"  {payload}x  " + (f"pass, beta:rate = {best}" if best else "FAIL"))
