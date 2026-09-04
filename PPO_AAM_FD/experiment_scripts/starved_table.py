"""Per-seed and per-mesh breakdown of the starved-encoder campaign, and the
LaTeX table bodies for Chapter 7.

The pooled percentage is what the pre-registered rule is decided on. The
per-mesh columns are printed beside it because seed 17's pooled cut-height
figure leans heavily on one mesh, and an aggregate that is carried by a single
condition is reported as such rather than as a uniform effect.
"""
import json
import math
import os
import random
import statistics as st

B = 20000
D = "outputs/rif_delay"
SEEDS = (17, 23, 101, 42, 7)
MESHES = ("2718", "3141", "8191")
KEYS = (("per_env_cut_error_m", "cut"), ("per_env_lateral_m", "lateral"))


def per_mesh(tag, key, mesh):
    f = f"{D}/{tag}_m{mesh}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    m = list(d["per_mesh"].values())[0]
    return [x * 1000.0 for x in m[key]]


def pooled(tag, key):
    v = []
    for mesh in MESHES:
        p = per_mesh(tag, key, mesh)
        if p is None:
            return None
        v += p
    return v


def stats(a, b, seed=0):
    """a = reference, b = arm. Positive => b better (lower error)."""
    random.seed(seed)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = [x - y for x, y in zip(a, b)]
    boot = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)])
                  for _ in range(B))
    lo, hi = boot[int(.025 * B)], boot[int(.975 * B)]
    lr = [math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0]
    pct = (math.exp(st.median(lr)) - 1) * 100 if lr else float("nan")
    return dict(n=n, pct=pct, med=st.median(diffs), lo=lo, hi=hi,
                sig=(lo > 0 or hi < 0))


print("=" * 78)
print("PRIMARY  control -> adaptive   (positive = live history better)")
print("=" * 78)
rows = {}
for key, name in KEYS:
    print(f"\n--- {name} ---")
    print(f"{'seed':>5}  {'pooled %':>10} {'paired med':>11} {'95% CI':>22}   per-mesh %")
    for s in SEEDS:
        c = pooled(f"starved{s}_control", key)
        a = pooled(f"starved{s}_adaptive", key)
        if c is None or a is None:
            print(f"{s:>5}  {'--- not yet scored ---':>10}")
            continue
        r = stats(c, a)
        pm = []
        for mesh in MESHES:
            cm = per_mesh(f"starved{s}_control", key, mesh)
            am = per_mesh(f"starved{s}_adaptive", key, mesh)
            lr = [math.log(x / y) for x, y in zip(cm, am) if x > 0 and y > 0]
            pm.append((math.exp(st.median(lr)) - 1) * 100)
        rows[(s, name)] = (r, pm)
        star = "*" if r["sig"] else " "
        print(f"{s:>5}  {r['pct']:+9.1f}%{star} {r['med']:+10.2f}mm "
              f"[{r['lo']:+7.2f},{r['hi']:+7.2f}]   "
              + "  ".join(f"{v:+7.1f}" for v in pm))

# Success rule, applied exactly as pre-registered.
BOUND = {"cut": 10.0, "lateral": 1.5}
print("\n" + "=" * 78)
print("PRE-REGISTERED SUCCESS RULE, per seed")
print("=" * 78)
passes = scored = 0
for s in SEEDS:
    if (s, "cut") not in rows:
        continue
    scored += 1
    res = {n: rows[(s, n)][0] for _, n in KEYS}
    wins = [n for n, r in res.items() if r["pct"] > 0 and r["sig"]]
    if len(wins) == 2:
        verdict, ok = "joint improvement", True
    elif len(wins) == 1:
        other = [n for n in res if n not in wins][0]
        r = res[other]
        ok = r["pct"] > 0 or not r["sig"] or abs(r["pct"]) <= BOUND[other]
        verdict = ("Pareto improvement" if ok
                   else f"trade-off past the {other} bound")
    else:
        verdict, ok = "no resolved improvement", False
    passes += ok
    print(f"  seed {s:>3}: {verdict}")
if scored:
    print(f"\n  {passes}/{scored} seeds pass the primary. "
          f"Median seed {'PASSES' if passes * 2 > scored else 'FAILS'}. "
          f"Declared must-have: >= 3/5.")
