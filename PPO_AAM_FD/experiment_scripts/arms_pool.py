"""Pre-registered analysis: adaptive (history) vs control (history zeroed).

Paired env-for-env within each mesh: identical seed, terrain and plant draw, so
env k is the same task in both arms and only the history differs. Pooled across
meshes. Bootstrap CIs on the paired median and the paired median log-ratio.

Pass condition, fixed in advance: the adaptive arm's CUT-HEIGHT error is lower
and the 95% CI on the paired median log-ratio EXCLUDES ZERO.
"""
import json, math, random, statistics as st, glob, os
random.seed(0); B = 20000

def load(arm, mesh):
    return json.load(open(f"outputs/rif_delay/arm_{arm}_m{mesh}.json"))

meshes = sorted({os.path.basename(f).split("_m")[1].split(".")[0]
                 for f in glob.glob("outputs/rif_delay/arm_*_m*.json")})
have = [m for m in meshes
        if os.path.exists(f"outputs/rif_delay/arm_history_m{m}.json")
        and os.path.exists(f"outputs/rif_delay/arm_nohistory_m{m}.json")]
print(f"meshes with BOTH arms: {have}")
if not have:
    raise SystemExit("need both arms on at least one mesh")

for metric, label in (("per_env_cut_error_m", "CUT HEIGHT (primary)"),
                      ("per_env_lateral_m", "LATERAL (secondary)")):
    adapt, ctrl = [], []
    for m in have:
        adapt += [x*1000 for x in load("history", m)[metric]]
        ctrl  += [x*1000 for x in load("nohistory", m)[metric]]
    n = len(adapt)
    diffs = [c - a for a, c in zip(adapt, ctrl)]        # >0 means adaptive BETTER
    o = st.median(diffs)
    s = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)]) for _ in range(B))
    lo, hi = s[int(.025*B)], s[int(.975*B)]
    lr = [math.log(a/c) for a, c in zip(adapt, ctrl) if a > 0 and c > 0]
    k = len(lr)
    sr = sorted(st.median([lr[random.randrange(k)] for _ in range(k)]) for _ in range(B))
    pc = lambda v: (math.exp(v)-1)*100
    print(f"\n=== {label}   pairs={n}")
    print(f"    adaptive (history live) median {st.median(adapt):8.3f} mm")
    print(f"    control  (history zero) median {st.median(ctrl):8.3f} mm")
    print(f"    paired median improvement {o:+8.3f} mm  CI[{lo:+8.3f},{hi:+8.3f}]")
    print(f"    adaptive vs control {pc(st.median(lr)):+7.1f}%  "
          f"CI[{pc(sr[int(.025*B)]):+7.1f},{pc(sr[int(.975*B)]):+7.1f}]")
    verdict = ("PASS -- CI excludes zero, adaptive better" if lo > 0 else
               "adaptive WORSE, CI excludes zero" if hi < 0 else
               "NULL -- CI includes zero")
    print(f"    -> {verdict}")
