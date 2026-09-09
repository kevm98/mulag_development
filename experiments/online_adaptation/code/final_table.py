"""The final table: 3 seeds x 3 masses x {Frozen, Oracle, Adaptive} + wrong-info.

Decision rule from PREREG_FINAL_SEEDS_2026-09-06, which is NOT a conjunction
over seeds:

    H1:  E_adaptive < E_frozen   under plant conditions where the frozen policy
                                 has a measurable robustness deficit

pooled and paired, with individual seeds shown transparently beside it. Two
seeds improving strongly and one neutral supports H1 if the pooled result is
convincing; one improving and two degrading does not, and is reported as
initialisation dependence.

Frozen is scored once per mass and reused across seeds: with the context zeroed
any trained checkpoint computes base(x) + W_ctx . 0 = base(x), so it IS the
frozen policy regardless of which seed trained the context.

Usage:  final_table.py [--dir outputs/final]
"""
import glob
import json
import math
import os
import random
import statistics as st
import sys

B = 20000
SEEDS = ("17", "23", "101")
MASSES = ("3.0", "4.0", "5.0")


def row(path):
    if not os.path.exists(path):
        return None
    for r in json.load(open(path)).get("per_mesh", {}).values():
        return r
    return None


def paired(ref, arm):
    random.seed(0)
    n = min(len(ref), len(arm))
    ref, arm = ref[:n], arm[:n]
    diffs = [x - y for x, y in zip(ref, arm)]
    boot = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)])
                  for _ in range(B))
    lo, hi = boot[int(0.025 * B)], boot[int(0.975 * B)]
    lr = [math.log(x / y) for x, y in zip(ref, arm) if x > 0 and y > 0]
    pct = (math.exp(st.median(lr)) - 1) * 100 if lr else float("nan")
    return pct, (lo > 0 or hi < 0), diffs


def mm(r, key):
    return st.median([v * 1000.0 for v in r[key]])


if __name__ == "__main__":
    root = (sys.argv[sys.argv.index("--dir") + 1]
            if "--dir" in sys.argv else "outputs/final")
    print("FINAL RESULT -- locked architecture, held-out mesh 3141")
    print("Frozen is the frozen policy exactly (context zeroed), shared across seeds.\n")

    for key, name in (("per_env_cut_error_m", "CUT HEIGHT"),
                      ("per_env_lateral_m", "LATERAL")):
        print(f"{'=' * 78}\n{name}\n{'=' * 78}")
        print(f"{'mass':>6} {'seed':>5} {'frozen':>9} {'adaptive':>9} "
              f"{'oracle':>9}   {'adaptive vs frozen':>20}")
        pooled = {m: [] for m in MASSES}
        for m in MASSES:
            fr = row(os.path.join(root, f"frozen_p{m}.json"))
            if fr is None:
                print(f"{m:>6}  frozen missing")
                continue
            for s in SEEDS:
                ad = row(os.path.join(root, f"adaptive_s{s}_p{m}.json"))
                orc = row(os.path.join(root, f"oracle_s{s}_p{m}.json"))
                if ad is None:
                    print(f"{m:>6} {s:>5}  adaptive missing")
                    continue
                pct, sig, diffs = paired(
                    [v * 1000 for v in fr[key]], [v * 1000 for v in ad[key]])
                pooled[m] += diffs
                oc = f"{mm(orc, key):9.2f}" if orc else "        -"
                print(f"{m:>6} {s:>5} {mm(fr, key):9.2f} {mm(ad, key):9.2f} "
                      f"{oc}   {pct:+18.1f}%{'*' if sig else ' '}")
            # Pooled across seeds at this mass.
            d = pooled[m]
            if d:
                random.seed(0)
                boot = sorted(st.median([d[random.randrange(len(d))]
                                         for _ in range(len(d))])
                              for _ in range(B))
                lo, hi = boot[int(0.025 * B)], boot[int(0.975 * B)]
                sig = lo > 0 or hi < 0
                print(f"{m:>6} {'POOL':>5} {'':>9} {'':>9} {'':>9}   "
                      f"median diff {st.median(d):+.2f} mm, CI "
                      f"[{lo:+.2f},{hi:+.2f}]{' *' if sig else ''}")
            print()

    # Wrong-information control.
    print(f"{'=' * 78}\nWRONG-INFORMATION CONTROL (mixed payload)\n{'=' * 78}")
    z = row(os.path.join(root, "wronginfo_zero.json"))
    lv = row(os.path.join(root, "wronginfo_live.json"))
    sh = row(os.path.join(root, "wronginfo_shuffle.json"))
    if all((z, lv, sh)):
        print(f"{'arm':>28} {'cut mm':>9} {'lateral mm':>11}")
        for n, r in (("frozen (no context)", z), ("true payload", lv),
                     ("payload SHUFFLED", sh)):
            print(f"{n:>28} {mm(r, 'per_env_cut_error_m'):9.2f} "
                  f"{mm(r, 'per_env_lateral_m'):11.2f}")
        for k, nm in (("per_env_cut_error_m", "cut"),
                      ("per_env_lateral_m", "lateral")):
            p1, s1, _ = paired([v * 1000 for v in z[k]], [v * 1000 for v in lv[k]])
            p2, s2, _ = paired([v * 1000 for v in z[k]], [v * 1000 for v in sh[k]])
            print(f"  {nm:>8}:  frozen->true {p1:+.1f}%{'*' if s1 else ''}   "
                  f"frozen->shuffled {p2:+.1f}%{'*' if s2 else ''}")
        print("\n  If true >> shuffled, the PAYLOAD INFORMATION is doing the work.")
        print("  If shuffled matches true, any context signal would do and the")
        print("  claim does not stand.")
    else:
        print("  (not yet scored)")
