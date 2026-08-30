"""Three-way paired comparison, all through the SAME evaluator.

    base      : frozen policy, residual OFF (alpha = 0)
    adaptive  : residual with history live
    control   : residual with history zeroed

The base arm is the one that decides what the adaptive/control gap MEANS:

  adaptive < base  -> adaptation genuinely improves on doing nothing
  adaptive ~ base  -> the latent makes the residual SAFE, but buys no gain
  adaptive > base  -> the residual costs more than it recovers, and the
                      adaptive-vs-control gap only measures how badly the
                      unguided residual breaks
"""
import json, math, random, statistics as st, glob, os
random.seed(0); B = 20000

def have(arm, m):
    return os.path.exists(f"outputs/rif_delay/arm_{arm}_m{m}.json")

meshes = sorted({os.path.basename(f).split("_m")[1].split(".")[0]
                 for f in glob.glob("outputs/rif_delay/arm_*_m*.json")})
usable = [m for m in meshes if all(have(a, m) for a in ("base", "history", "nohistory"))]
print(f"meshes with all three arms: {usable}")
if not usable:
    partial = [m for m in meshes if have("base", m)]
    print(f"(base available on: {partial})")
    raise SystemExit(0)

def pool(arm, metric):
    out = []
    for m in usable:
        out += [x*1000 for x in json.load(open(f"outputs/rif_delay/arm_{arm}_m{m}.json"))[metric]]
    return out

def paired(x, y, label):
    """log-ratio of y relative to x; positive means y WORSE than x."""
    lr = [math.log(b/a) for a, b in zip(x, y) if a > 0 and b > 0]
    n = len(lr)
    s = sorted(st.median([lr[random.randrange(n)] for _ in range(n)]) for _ in range(B))
    pc = lambda v: (math.exp(v)-1)*100
    o, lo, hi = pc(st.median(lr)), pc(s[int(.025*B)]), pc(s[int(.975*B)])
    sig = "" if lo < 0 < hi else "  <== EXCLUDES ZERO"
    print(f"    {label:34s} {o:+8.1f}%  CI[{lo:+8.1f},{hi:+8.1f}]{sig}")
    return o, lo, hi

for metric, name in (("per_env_cut_error_m", "CUT HEIGHT (primary)"),
                     ("per_env_lateral_m", "LATERAL (secondary)")):
    base, adapt, ctrl = pool("base", metric), pool("history", metric), pool("nohistory", metric)
    print(f"\n=== {name}   pairs={len(base)}")
    print(f"    base (residual OFF)      median {st.median(base):9.3f} mm")
    print(f"    adaptive (history live)  median {st.median(adapt):9.3f} mm")
    print(f"    control  (history zero)  median {st.median(ctrl):9.3f} mm")
    print()
    _, lo_a, _ = paired(base, adapt, "adaptive vs base")
    paired(base, ctrl, "control vs base")
    paired(adapt, ctrl, "control vs adaptive")
    if metric == "per_env_cut_error_m":
        print("\n    Read: 'adaptive vs base' is the claim. Negative and excluding")
        print("    zero means adaptation beats doing nothing. If it does NOT exclude")
        print("    zero, the adaptive-vs-control gap only measures residual damage.")
