"""The multi-seed triplet table.

Design and success criterion are fixed in reports/PREREG_TRIPLET_2026-09-01.md
and are NOT re-decided here.

  base    -> control    does extra capacity alone help?      (prediction 1)
  control -> adaptive   does LIVE HISTORY help beyond it?    (PRIMARY)
  base    -> adaptive   context; already-seen axis, caveated

SUCCESS = one metric improves significantly while the other improves or stays
within its measured non-inferiority bound (cut 10%, lateral 1.5%; A/A floors
from PREREG_R2_2026-09-01.md).

Usage:  triplet.py [--ckpt best|399] [--meshes 3|5]
"""
import json, os, sys, math, random, statistics as st

B = 20000
D = "outputs/rif_delay"
BOUND = {"cut": 10.0, "lateral": 1.5}
KEYS = (("per_env_cut_error_m", "cut"), ("per_env_lateral_m", "lateral"))
SEEDS = (17, 23, 101, 42, 7)


def pool(tag, key, meshes):
    v = []
    for m in meshes:
        f = f"{D}/{tag}_m{m}.json"
        if not os.path.exists(f):
            return None
        d = json.load(open(f))
        for _, r in d["per_mesh"].items():
            v += [x * 1000.0 for x in r[key]]
    return v


def compare(a, b):
    """a = reference, b = arm. Positive pct => b is better (smaller error)."""
    random.seed(0)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    diffs = [x - y for x, y in zip(a, b)]
    boot = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)])
                  for _ in range(B))
    lo, hi = boot[int(.025 * B)], boot[int(.975 * B)]
    lr = [math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0]
    pct = (math.exp(st.median(lr)) - 1) * 100
    return dict(n=n, pct=pct, lo=lo, hi=hi, sig=(lo > 0 or hi < 0),
                med_a=st.median(a), med_b=st.median(b), diff=st.median(diffs))


def label(res):
    """Apply the pre-registered success criterion to a {cut, lateral} pair."""
    cut, lat = res["cut"], res["lateral"]
    wins = [n for n, r in res.items() if r["pct"] > 0 and r["sig"]]
    if not wins:
        if all(r["pct"] < 0 and r["sig"] for r in res.values()):
            return "both worse", False
        return "no significant improvement", False
    if len(wins) == 2:
        return "joint improvement", True
    other = "lateral" if wins[0] == "cut" else "cut"
    o = res[other]
    if o["pct"] >= -BOUND[other]:
        return f"Pareto improvement ({other} maintained)", True
    return f"trade-off ({other} worse)", False


def cell(r):
    return f"{r['pct']:+7.1f}%{'*' if r['sig'] else ' '}"


def run(ref, arm, meshes):
    res = {}
    for key, name in KEYS:
        a, b = pool(ref, key, meshes), pool(arm, key, meshes)
        if a is None or b is None:
            return None, (ref if a is None else arm)
        res[name] = compare(a, b)
    return res, None


def table(title, pairs, meshes, verdict=True):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"{'seed':>5}  {'n':>5}  {'cut':>9}  {'lateral':>9}   outcome")
    passes = []
    for seed, ref, arm in pairs:
        res, missing = run(ref, arm, meshes)
        if res is None:
            print(f"{seed:>5}  {'':>5}  MISSING: {missing}")
            passes.append(None); continue
        lab, ok = label(res)
        n = res["cut"]["n"]
        print(f"{seed:>5}  {n:>5}  {cell(res['cut'])}  {cell(res['lateral'])}   {lab}")
        passes.append(ok)
    if verdict:
        got = [p for p in passes if p is not None]
        if len(got) < len(pairs):
            print(f"\n  INCOMPLETE: {len(got)}/{len(pairs)} seeds scored.")
        else:
            k = sum(got)
            med = "PASSES" if k >= (len(got) + 1) // 2 else "FAILS"
            print(f"\n  {k}/{len(got)} seeds succeed -> the MEDIAN SEED {med}.")
    return passes


if __name__ == "__main__":
    ck = sys.argv[sys.argv.index("--ckpt") + 1] if "--ckpt" in sys.argv else "best"
    nm = sys.argv[sys.argv.index("--meshes") + 1] if "--meshes" in sys.argv else "5"
    meshes = ("1337", "17", "4242") + (("909", "7001") if nm == "5" else ())
    if ck == "best":
        adap = {s: (f"rebal101_best250" if s == 101 else f"rebal{s}_best") for s in SEEDS}
        ctrl = {s: f"rebal{s}_nohist_best" for s in SEEDS}
        which = "R3 best-by-training-reward checkpoint (PRIMARY)"
    else:
        adap = {s: f"rebal{s}_history" for s in SEEDS}
        ctrl = {s: f"rebal{s}_nohistory" for s in SEEDS}
        which = "model_399, final checkpoint (SECONDARY / robustness)"

    print(f"Checkpoint protocol: {which}")
    print(f"Meshes: {', '.join(meshes)}.  Paired per-env differences, "
          f"median, bootstrap CI B={B}. * = CI excludes zero.")
    print(f"Non-inferiority bounds: cut {BOUND['cut']}%, lateral {BOUND['lateral']}% "
          f"(A/A floors, PREREG_R2).")

    table("PRIMARY - capacity-matched control -> adaptive  (does LIVE HISTORY help?)",
          [(s, ctrl[s], adap[s]) for s in SEEDS], meshes)
    table("PREDICTION 1 - base -> capacity-matched control  (does CAPACITY alone help?)",
          [(s, "base_R1", ctrl[s]) for s in SEEDS], meshes, verdict=False)
    table("CONTEXT - base -> adaptive  (already-seen axis; 4th look, see prereg)",
          [(s, "base_R1", adap[s]) for s in SEEDS], meshes)
