"""Three-arm comparison through the VALIDATED gate path.

    base      alpha=0             frozen policy
    adaptive  alpha=0.5, history  can sense the mid-episode flip
    control   alpha=0.5, zeroed   same weights, cannot sense

RMA's claim, and the one that matters here, is that online adaptation IMPROVES
task performance over non-adaptive baselines under changing dynamics. So the
bar is BOTH:

    adaptive < control   (the latent is doing work)
    adaptive < base      (the whole thing beats doing nothing)

with CIs excluding zero on cut-height error. adaptive ~ base would show only
that the latent keeps the residual safe -- a footnote, not the claim.
"""
import json, math, random, statistics as st, glob, os
random.seed(0); B = 20000
ARMS = ("base", "adaptive", "control")

meshes = sorted({os.path.basename(f).split("_m")[1].split(".")[0]
                 for f in glob.glob("outputs/rif_delay/g_*_m*.json")})
usable = [m for m in meshes
          if all(os.path.exists(f"outputs/rif_delay/g_{a}_m{m}.json") for a in ARMS)]
print(f"meshes with all three arms: {usable}")
if not usable:
    for a in ARMS:
        got = [m for m in meshes if os.path.exists(f"outputs/rif_delay/g_{a}_m{m}.json")]
        print(f"  {a}: {got}")
    raise SystemExit(0)

def pool(arm, metric):
    out = []
    for m in usable:
        r = json.load(open(f"outputs/rif_delay/g_{arm}_m{m}.json"))["per_mesh"][m]
        out += [x * 1000 for x in r[metric]]
    return out

def cmp(x, y, label):
    lr = [math.log(b / a) for a, b in zip(x, y) if a > 0 and b > 0]
    n = len(lr)
    s = sorted(st.median([lr[random.randrange(n)] for _ in range(n)]) for _ in range(B))
    pc = lambda v: (math.exp(v) - 1) * 100
    o, lo, hi = pc(st.median(lr)), pc(s[int(.025*B)]), pc(s[int(.975*B)])
    tag = "" if lo < 0 < hi else "  <== EXCLUDES ZERO"
    print(f"    {label:32s} {o:+8.1f}%  CI[{lo:+8.1f},{hi:+8.1f}]{tag}")
    return lo, hi

for metric, name in (("per_env_cut_error_m", "CUT HEIGHT (primary)"),
                     ("per_env_lateral_m", "LATERAL (secondary)")):
    base, adapt, ctrl = (pool(a, metric) for a in ARMS)
    print(f"\n=== {name}   pairs={len(base)}")
    for label, v in (("base (frozen)", base), ("adaptive", adapt), ("control", ctrl)):
        print(f"    {label:16s} median {st.median(v):9.3f} mm")
    print()
    lo_ab, hi_ab = cmp(adapt, base, "base vs adaptive")
    lo_ac, hi_ac = cmp(adapt, ctrl, "control vs adaptive")
    if metric == "per_env_cut_error_m":
        beats_base = lo_ab > 0
        beats_ctrl = lo_ac > 0
        print()
        if beats_base and beats_ctrl:
            print("    >>> RMA-STYLE PASS: adaptation improves on BOTH the frozen")
            print("        base and the non-adaptive control.")
        elif beats_ctrl:
            print("    >>> PARTIAL: the latent beats the non-adaptive control, but")
            print("        NOT the frozen base -- safety, not improvement.")
        else:
            print("    >>> NULL on the primary metric.")
