"""The two comparisons that constitute the result. Nothing else.

  1. base vs adaptive        -- adaptation helps, on BOTH metrics
  2. adaptive vs no-history  -- the benefit comes from the online estimate,
                                not from the extra parameters

Both are paired over the same environments and meshes, reported as median with a
bootstrap CI on the paired difference. An arm PASSES only if BOTH cut-height and
lateral error improve with the CI excluding zero: a win on one metric and a loss
on the other is a trade-off and counts as a failure.

Usage:  verdict.py <adaptive_tag> [control_tag] [--base BASE_TAG]
        tags are the outputs/rif_delay/<tag>_m<mesh>.json prefixes
"""
import json, os, sys, math, random, statistics as st

random.seed(0)
B = 20000
MESHES = ("1337", "17", "4242")
D = "outputs/rif_delay"

def pool(tag, key):
    v = []
    for m in MESHES:
        f = f"{D}/{tag}_m{m}.json"
        if os.path.exists(f):
            d = json.load(open(f))
            for _, r in d["per_mesh"].items():
                v += [x * 1000.0 for x in r[key]]
    return v

def compare(a, b):
    """a = reference (worse expected), b = arm. Positive % => b is better."""
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    diffs = [x - y for x, y in zip(a, b)]
    med = st.median(diffs)
    boot = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)]) for _ in range(B))
    lo, hi = boot[int(.025 * B)], boot[int(.975 * B)]
    lr = [math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0]
    pcm = (math.exp(st.median(lr)) - 1) * 100
    return dict(n=n, med_a=st.median(a), med_b=st.median(b), diff=med,
                lo=lo, hi=hi, pct=pcm, sig=(lo > 0 or hi < 0))

def row(label, r):
    mark = "*" if r["sig"] else " "
    print(f"  {label:22s} {r['med_a']:8.3f} -> {r['med_b']:8.3f} mm   "
          f"{r['pct']:+7.1f}%{mark}   paired {r['diff']:+7.3f} [{r['lo']:+7.3f},{r['hi']:+7.3f}]")

def block(title, ref_tag, arm_tag):
    print(f"\n{title}")
    verdicts = {}
    for key, name in (("per_env_cut_error_m", "cut height"), ("per_env_lateral_m", "lateral")):
        a, b = pool(ref_tag, key), pool(arm_tag, key)
        if not a or not b:
            print(f"  {name}: MISSING data ({ref_tag} or {arm_tag})"); return None
        r = compare(a, b); row(name, r); verdicts[name] = r
    both = all(v["pct"] > 0 and v["sig"] for v in verdicts.values())
    worse = all(v["pct"] < 0 and v["sig"] for v in verdicts.values())
    print(f"  => {'PASS - both metrics improve' if both else ('both worse' if worse else 'TRADE-OFF (fails the bar)')}")
    return both

if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    base = "g_base"
    if "--base" in sys.argv:
        base = sys.argv[sys.argv.index("--base") + 1]
    adaptive = args[0]
    control = args[1] if len(args) > 1 else None
    print(f"n=528 paired environments, meshes {', '.join(MESHES)}. "
          f"* = bootstrap CI on the paired difference excludes zero.")
    ok = block(f"EXPERIMENT 1  base ({base}) vs adaptive ({adaptive})", base, adaptive)
    if control:
        block(f"EXPERIMENT 2  history-zeroed control ({control}) vs adaptive ({adaptive})", control, adaptive)
    sys.exit(0 if ok else 1)
