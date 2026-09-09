"""Second held-out mesh: does the payload result repeat on 8191?

Same statistic as the main table, paired per environment, so the two meshes are
directly comparable. Mesh 8191 was reserved and never used to design, diagnose
or read this campaign.
"""
import json, os, random, statistics as st, sys

B = 20000
SEEDS = ("17", "23", "101")
MASSES = ("4.0", "5.0")
ROOT = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "outputs/heldout8191"


def row(p):
    f = os.path.join(ROOT, p)
    if not os.path.exists(f):
        return None
    d = json.load(open(f)).get("per_mesh", {})
    return list(d.values())[0] if d else None


def paired(fr, ad, key, unit, hb):
    f = [v * unit for v in fr[key]]
    a = [v * unit for v in ad[key]]
    n = min(len(f), len(a))
    d = [(a[i] - f[i]) if hb else (f[i] - a[i]) for i in range(n)]
    random.seed(0)
    bs = sorted(st.median([d[random.randrange(n)] for _ in range(n)])
                for _ in range(B))
    return st.median(d), bs[int(.025 * B)], bs[int(.975 * B)], d, st.median(f), st.median(a)


print("MESH 8191 -- independent repetition of the payload result\n")
for key, name, unit, hb in (("per_env_cut_error_m", "CUT HEIGHT (mm)", 1000.0, False),
                            ("per_env_lateral_m", "LATERAL (mm)", 1000.0, False),
                            ("per_env_contact_fraction", "GROUND CONTACT", 1.0, True)):
    print("=" * 78); print(name); print("=" * 78)
    print(f"{'mass':>5} {'seed':>5} {'frozen':>9} {'adaptive':>9} {'median diff':>12} {'95% CI':>22} {'better':>8}")
    for m in MASSES:
        fr = row(f"frozen_p{m}.json")
        if fr is None:
            print(f"{m:>5}  frozen missing"); continue
        pool = []
        for s in SEEDS:
            ad = row(f"adaptive_s{s}_p{m}.json")
            if ad is None:
                print(f"{m:>5} {s:>5}  missing"); continue
            md, lo, hi, d, mf, ma = paired(fr, ad, key, unit, hb)
            pool += d
            sig = "*" if (lo > 0 or hi < 0) else " "
            win = 100.0 * sum(1 for v in d if v > 0) / len(d)
            print(f"{m:>5} {s:>5} {mf:9.2f} {ma:9.2f} {md:+12.2f} [{lo:+9.2f},{hi:+9.2f}]{sig} {win:7.1f}%")
        if pool:
            random.seed(0)
            n = len(pool)
            bs = sorted(st.median([pool[random.randrange(n)] for _ in range(n)]) for _ in range(B))
            lo, hi = bs[int(.025 * B)], bs[int(.975 * B)]
            sig = "*" if (lo > 0 or hi < 0) else " "
            win = 100.0 * sum(1 for v in pool if v > 0) / n
            print(f"{m:>5} {'POOL':>5} {'':>9} {'':>9} {st.median(pool):+12.2f} [{lo:+9.2f},{hi:+9.2f}]{sig} {win:7.1f}%")
        print()
print("Positive = adaptive better. * = 95% CI excludes zero.")
