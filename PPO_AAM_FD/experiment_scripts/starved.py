"""Score the starved-encoder campaign under the rule fixed in
reports/PREREG_STARVED_2026-09-03.md.

PRIMARY   control  -> adaptive   does live history help beyond equal capacity?
SECONDARY base     -> adaptive   already-seen axis, cannot separate the two
CONTEXT   base     -> control    does extra capacity alone help?

SUCCESS = one channel improves with its paired bootstrap interval excluding
zero, and the other improves or stays within its non-inferiority bound
(cut 10 %, lateral 1.5 %). The campaign is positive if the MEDIAN SEED passes
the PRIMARY comparison. The rule is not re-decided here.
"""
import json
import math
import os
import random
import statistics as st
import sys

B = 20000
D = "outputs/rif_delay"
BOUND = {"cut": 10.0, "lateral": 1.5}
KEYS = (("per_env_cut_error_m", "cut"), ("per_env_lateral_m", "lateral"))
SEEDS = (17, 23, 101, 42, 7)
MESHES = ("2718", "3141", "8191")


def pool(tag, key):
    v = []
    for m in MESHES:
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
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = [x - y for x, y in zip(a, b)]
    boot = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)])
                  for _ in range(B))
    lo, hi = boot[int(.025 * B)], boot[int(.975 * B)]
    lr = [math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0]
    pct = (math.exp(st.median(lr)) - 1) * 100 if lr else float("nan")
    return dict(n=n, pct=pct, lo=lo, hi=hi, sig=(lo > 0 or hi < 0),
                med_a=st.median(a), med_b=st.median(b))


def label(res):
    """The pre-registered success criterion on a {cut, lateral} pair."""
    wins = [n for n, r in res.items() if r["pct"] > 0 and r["sig"]]
    if not wins:
        if all(r["pct"] < 0 and r["sig"] for r in res.values()):
            return "both worse", False
        return "no significant improvement", False
    if len(wins) == 2:
        return "joint improvement", True
    other = [n for n in res if n not in wins][0]
    r = res[other]
    if r["pct"] > 0 or not r["sig"] or abs(r["pct"]) <= BOUND[other]:
        return f"Pareto improvement ({wins[0]} resolved, {other} within bound)", True
    return f"trade-off ({wins[0]} up, {other} down past bound)", False


def run(ref_tag, arm_tag, title):
    print(f"\n########## {title}:  {ref_tag} -> {arm_tag}")
    passes = 0
    scored = 0
    for s in SEEDS:
        res = {}
        ok = True
        for key, name in KEYS:
            a = pool(ref_tag.format(s=s), key)
            b = pool(arm_tag.format(s=s), key)
            if a is None or b is None:
                ok = False
                break
            res[name] = compare(a, b)
        if not ok:
            print(f"  seed {s:3d}: MISSING")
            continue
        scored += 1
        txt, good = label(res)
        passes += good
        c, l = res["cut"], res["lateral"]
        print(f"  seed {s:3d}: cut {c['pct']:+7.1f}%{'*' if c['sig'] else ' '}"
              f"  lateral {l['pct']:+7.1f}%{'*' if l['sig'] else ' '}"
              f"   n={c['n']:4d}   {txt}")
    if scored:
        verdict = "PASSES" if passes * 2 > scored else "FAILS"
        print(f"  --> {passes}/{scored} seeds succeed; the median seed {verdict}")
    return passes, scored


if __name__ == "__main__":
    run("starvedbase", "starved{s}_control", "CONTEXT   capacity alone")
    run("starvedbase", "starved{s}_adaptive", "SECONDARY vs frozen base")
    p, n = run("starved{s}_control", "starved{s}_adaptive",
               "PRIMARY   live history at equal capacity")
    if n:
        print(f"\nCAMPAIGN VERDICT: primary succeeds on {p}/{n} seeds. "
              f"Declared must-have was >= 3/5.")
