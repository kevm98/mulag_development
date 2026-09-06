"""The held-out confirmation table.

Mesh 2718 is where the context-only campaign was designed, diagnosed and read,
so its passes are not confirmatory. 3141 and 8191 are the reserved meshes. The
cautionary case is on file: the online-gradient investigation had a favourable
rung on 2718 that reversed to -79.6% cut on 3141, and that reversal closed it.

Arms: ctxzero (the frozen policy, by construction, since the trained checkpoint
with its context zeroed computes base(x) + W_ctx . 0) and ctxest (the payload
estimated online). Statistic and bounds are triplet.py's, unchanged.

Ground contact is reported as a paired median difference in points, because the
cut-height channel carries a ~40% run-to-run floor on this instrument while
contact carries about 3%, and contact is the same physics: cut_height_error is
relu(h - reference) and exactly zero in contact.

Usage:  heldout_table.py [--dir outputs/heldout]
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


def row(path):
    for r in json.load(open(path)).get("per_mesh", {}).values():
        return r
    return None


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


def gain(ref, arm):
    random.seed(0)
    n = min(len(ref), len(arm))
    diffs = [y - x for x, y in zip(ref[:n], arm[:n])]
    boot = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)])
                  for _ in range(B))
    return dict(delta=st.median(diffs),
                sig=(boot[int(0.025 * B)] > 0 or boot[int(0.975 * B)] < 0))


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
            if "--dir" in sys.argv else "outputs/heldout")
    keys = sorted({tuple(os.path.basename(f)[:-5].split("_")[1:])
                   for f in glob.glob(os.path.join(root, "ctx*_p*_m*.json"))})
    if not keys:
        raise SystemExit(f"no reports under {root}")
    print("HELD-OUT CONFIRMATION. Meshes 3141 and 8191 were never used to "
          "design, diagnose\nor read this campaign. Paired per-environment "
          f"differences, B={B}. * = CI excludes zero.")
    print(f"\n{'mesh':>6} {'payload':>8}  {'arm':>8}  {'cut mm':>9} {'lat mm':>9} "
          f"{'contact':>8} {'safety':>7}")
    verdicts = []
    for p, m in keys:
        z = os.path.join(root, f"ctxzero_{p}_{m}.json")
        e = os.path.join(root, f"ctxest_{p}_{m}.json")
        if not (os.path.exists(z) and os.path.exists(e)):
            continue
        rz, re_ = row(z), row(e)
        mesh, pay = m[1:], p[1:]
        for name, r in (("ctxzero", rz), ("ctxest", re_)):
            print(f"{mesh:>6} {pay:>8}  {name:>8}  "
                  f"{st.median([v * 1000 for v in r['per_env_cut_error_m']]):9.2f} "
                  f"{st.median([v * 1000 for v in r['per_env_lateral_m']]):9.2f} "
                  f"{st.mean(r['per_env_contact_fraction']):8.3f} "
                  f"{r['safety_violation_rate']:7.3f}")
        res = {n: compare([v * 1000 for v in rz[k]], [v * 1000 for v in re_[k]])
               for k, n in (("per_env_cut_error_m", "cut"),
                            ("per_env_lateral_m", "lateral"))}
        con = gain(rz["per_env_contact_fraction"], re_["per_env_contact_fraction"])
        lab, ok = label(res)
        verdicts.append((mesh, pay, lab, ok, res, con))

    print(f"\n{'mesh':>6} {'payload':>8}  {'cut':>10} {'lateral':>10} "
          f"{'contact':>9}   outcome")
    for mesh, pay, lab, ok, res, con in verdicts:
        cell = lambda r: f"{r['pct']:+9.1f}%{'*' if r['sig'] else ' '}"
        print(f"{mesh:>6} {pay:>8}  {cell(res['cut'])} {cell(res['lateral'])} "
              f"{con['delta']:+8.3f}{'*' if con['sig'] else ' '}  {lab}")
    passed = sum(1 for *_x, ok, _r, _c in verdicts if ok)
    print(f"\n  {passed}/{len(verdicts)} held-out cells pass the pre-registered rule.")
    if verdicts and passed == len(verdicts):
        print("  The selection-mesh result reproduces where it was not designed.")
    elif passed:
        print("  PARTIAL. Report which cells reproduce and which do not; do not "
              "pool them.")
    else:
        print("  It does NOT reproduce off the selection mesh. The 2718 passes "
              "are not claimable.")
