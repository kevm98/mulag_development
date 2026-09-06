"""The context-only campaign table.

Arms, all the SAME 70-wide policy; only where the payload number comes from
changes:

  base      the frozen distilled policy
  ctxzero   the trained checkpoint with its context zeroed. By construction this
            computes base(x) + W_ctx . 0 = base(x), so it should reproduce base.
            It is scored rather than asserted: if it does NOT match base, the
            freeze leaked and every other number here is suspect.
  ctxlive   the trained checkpoint reading the TRUE payload. Privileged, the
            ceiling.
  ctxest    the trained checkpoint reading a payload estimated ONLINE from the
            task observation. The deployable arm.

Comparisons:

  base -> ctxlive   does payload conditioning help a frozen policy at all?
  base -> ctxest    does an ONLINE estimate help? This is the deployable claim.
  ctxlive vs ctxest how much of the privileged ceiling the estimate recovers.
  base vs ctxzero   construction check, expected to be a null.

Statistic and bounds are triplet.py's, unchanged. Secondary outcomes are the
root-cause quantities: contact, hard impacts, tool acceleration, safety.

Usage:  ctxonly_table.py [--dir outputs/ctxonly] [--mesh 2718]
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
ARMS = ("base", "ctxzero", "ctxlive", "ctxest")


def load(root, arm, payload, mesh):
    path = os.path.join(root, f"{arm}_p{payload}_m{mesh}.json")
    if not os.path.exists(path):
        return None
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
    return dict(pct=pct, lo=lo, hi=hi, sig=(lo > 0 or hi < 0))


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


def pair(ref_row, arm_row):
    return {
        name: compare([v * 1000.0 for v in ref_row[key]],
                      [v * 1000.0 for v in arm_row[key]])
        for key, name in (("per_env_cut_error_m", "cut"),
                          ("per_env_lateral_m", "lateral"))
    }


def run(root, mesh, payload):
    rows = {a: load(root, a, payload, mesh) for a in ARMS}
    have = {a: r for a, r in rows.items() if r is not None}
    if "base" not in have or "ctxlive" not in have:
        print(f"\npayload {payload}x: waiting on {sorted(set(ARMS) - set(have))}")
        return None

    print(f"\n{'=' * 104}")
    print(f"payload {payload}x, mesh {mesh}")
    print('=' * 104)
    print(f"{'arm':>10}  {'cut mm':>9} {'lat mm':>9}  {'contact':>8} "
          f"{'impact':>7} {'accel':>7} {'safety':>7}")
    for a in ARMS:
        r = have.get(a)
        if r is None:
            continue
        print(f"{a:>10}  "
              f"{st.median([v * 1000 for v in r['per_env_cut_error_m']]):9.2f} "
              f"{st.median([v * 1000 for v in r['per_env_lateral_m']]):9.2f}  "
              f"{st.mean(r['per_env_contact_fraction']):8.3f} "
              f"{r['hard_impact_step_fraction']:7.3f} "
              f"{r['p95_tool_acceleration_mps2']:7.1f} "
              f"{r['safety_violation_rate']:7.3f}")

    cell = lambda r: f"{r['pct']:+9.1f}%{'*' if r['sig'] else ' '}"
    print(f"\n{'comparison':>34}  {'cut':>10}  {'lateral':>10}   outcome")
    out = {}
    for ref, arm, tag in (
        ("base", "ctxzero", "CHECK     base -> ctxzero (null?)"),
        ("base", "ctxlive", "CEILING   base -> ctxlive (oracle)"),
        ("base", "ctxest", "DEPLOY    base -> ctxest (online)"),
        ("ctxlive", "ctxest", "GAP       ctxlive -> ctxest"),
    ):
        if ref not in have or arm not in have:
            continue
        res = pair(have[ref], have[arm])
        lab, ok = label(res)
        print(f"{tag:>34}  {cell(res['cut'])}  {cell(res['lateral'])}   {lab}")
        out[(ref, arm)] = (ok, res)

    chk = out.get(("base", "ctxzero"))
    if chk and (chk[1]["cut"]["sig"] or chk[1]["lateral"]["sig"]):
        print("\n  WARNING: base -> ctxzero is NOT a null. The freeze leaked, or "
              "the zeroed\n  context is not reproducing the frozen policy. Treat "
              "the other rows as suspect.")
    return out


if __name__ == "__main__":
    root = (sys.argv[sys.argv.index("--dir") + 1]
            if "--dir" in sys.argv else "outputs/ctxonly")
    mesh = (sys.argv[sys.argv.index("--mesh") + 1]
            if "--mesh" in sys.argv else "2718")
    found = sorted({os.path.basename(f).split("_p")[1].split("_m")[0]
                    for f in glob.glob(os.path.join(root, f"*_p*_m{mesh}.json"))
                    if "_pmix_" not in f}, key=float)
    if not found:
        raise SystemExit(f"no reports under {root} for mesh {mesh}")
    print(f"Context-only conditioning, mesh {mesh}. Paired per-environment "
          f"differences, B={B}.\n* = CI excludes zero. Bounds: cut "
          f"{BOUND['cut']}%, lateral {BOUND['lateral']}%.")
    results = {p: run(root, mesh, p) for p in found}

    print(f"\n{'=' * 104}\nVERDICT\n{'=' * 104}")
    for payload, out in results.items():
        if not out:
            print(f"  {payload}x  incomplete")
            continue
        for key, tag in ((("base", "ctxlive"), "oracle"),
                         (("base", "ctxest"), "ONLINE")):
            if key in out:
                ok, res = out[key]
                print(f"  {payload}x {tag:>7}  {'PASS' if ok else 'fail'}  "
                      f"cut {res['cut']['pct']:+.1f}%  "
                      f"lateral {res['lateral']['pct']:+.1f}%")
