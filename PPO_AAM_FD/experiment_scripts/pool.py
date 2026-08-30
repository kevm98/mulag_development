import json, math, random, statistics as st, glob, os
random.seed(0); B = 20000
ARMS = ["privileged", "mean", "shuffled", "zero"]
files = sorted(glob.glob("outputs/rif_delay/gate_m*.json"))
print(f"meshes: {[os.path.basename(f) for f in files]}")
pooled = {m: {a: [] for a in ARMS} for m in ("per_env_cut_error_m", "per_env_lateral_m")}
CY = None
for f in files:
    d = json.load(open(f))
    for mesh, r in d["per_mesh"].items():
        n = len(r["per_env_lateral_m"])
        CY = n // len(ARMS)
        assert n % len(ARMS) == 0
        for metric in pooled:
            v = [x * 1000 for x in r[metric]]
            for b, a in enumerate(ARMS):
                pooled[metric][a].extend(v[b*CY:(b+1)*CY])
print(f"block per arm per mesh = {CY}")
for metric, label in (("per_env_cut_error_m","CUT HEIGHT"), ("per_env_lateral_m","LATERAL")):
    P = pooled[metric]; n = len(P["privileged"])
    print(f"\n=== {label}  pairs={n}")
    for a in ARMS: print(f"    {a:11s} median {st.median(P[a]):8.3f} mm")
    base = P["privileged"]
    for a in ARMS[1:]:
        arm = P[a]
        diffs = [x-y for x,y in zip(base,arm)]
        o = st.median(diffs)
        s = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)]) for _ in range(B))
        lo,hi = s[int(.025*B)], s[int(.975*B)]
        lr = [math.log(y/x) for x,y in zip(base,arm) if x>0 and y>0]; m=len(lr)
        sr = sorted(st.median([lr[random.randrange(m)] for _ in range(m)]) for _ in range(B))
        pc = lambda v:(math.exp(v)-1)*100
        sig = "  <== EXCLUDES ZERO" if (lo>0 or hi<0) else ""
        print(f"    privileged vs {a:9s}: paired median {o:+7.3f} mm CI[{lo:+7.3f},{hi:+7.3f}] | "
              f"{a} {pc(st.median(lr)):+6.1f}% [{pc(sr[int(.025*B)]):+6.1f},{pc(sr[int(.975*B)]):+6.1f}]{sig}")
