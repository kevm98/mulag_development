"""Per-seed paired statistics that survive a bimodal paired distribution.

The median log-ratio reported by final_table.py disagrees in SIGN with the
marginal medians in several cells (seed 23, 5.0x cut: medians 267 -> 111 mm but
ratio -27.3%). That is not a bug in either number: with an even sample and a
two-lobed paired distribution, median(x-y) and median(x/y) can land on opposite
sides of their respective null values. For a heterogeneous effect the paired
median DIFFERENCE and the fraction of environments improved are the honest
summaries, so both are reported here alongside the marginal medians.
"""
import glob, json, math, os, random, statistics as st

B = 20000
SEEDS = ("17", "23", "101")
MASSES = ("3.0", "4.0", "5.0")
ROOT = "outputs/final"


def row(p):
    if not os.path.exists(p):
        return None
    d = json.load(open(p)).get("per_mesh", {})
    return list(d.values())[0] if d else None


def boot_ci(d):
    random.seed(0)
    n = len(d)
    bs = sorted(st.median([d[random.randrange(n)] for _ in range(n)])
                for _ in range(B))
    return bs[int(0.025 * B)], bs[int(0.975 * B)]


for key, name, unit in (("per_env_cut_error_m", "CUT HEIGHT", 1000.0),
                        ("per_env_lateral_m", "LATERAL", 1000.0),
                        ("per_env_contact_fraction", "GROUND CONTACT", 1.0)):
    # For contact, HIGHER is better, so the improvement sign flips.
    higher_better = key == "per_env_contact_fraction"
    print("=" * 92)
    print(f"{name}   (paired per environment, held-out mesh 3141, n=176)")
    print("=" * 92)
    print(f"{'mass':>5} {'seed':>5} {'frozen':>9} {'adaptive':>9} "
          f"{'median diff':>13} {'95% CI':>22} {'envs better':>12}")
    for m in MASSES:
        fr = row(f"{ROOT}/frozen_p{m}.json")
        pool = []
        for s in SEEDS:
            ad = row(f"{ROOT}/adaptive_s{s}_p{m}.json")
            if fr is None or ad is None:
                continue
            f_ = [v * unit for v in fr[key]]
            a_ = [v * unit for v in ad[key]]
            n = min(len(f_), len(a_))
            d = [(a_[i] - f_[i]) if higher_better else (f_[i] - a_[i])
                 for i in range(n)]
            pool += d
            lo, hi = boot_ci(d)
            sig = "*" if (lo > 0 or hi < 0) else " "
            win = 100.0 * sum(1 for v in d if v > 0) / n
            print(f"{m:>5} {s:>5} {st.median(f_):9.2f} {st.median(a_):9.2f} "
                  f"{st.median(d):+13.2f} [{lo:+8.2f},{hi:+8.2f}]{sig} {win:11.1f}%")
        if pool:
            lo, hi = boot_ci(pool)
            sig = "*" if (lo > 0 or hi < 0) else " "
            win = 100.0 * sum(1 for v in pool if v > 0) / len(pool)
            print(f"{m:>5} {'POOL':>5} {'':>9} {'':>9} "
                  f"{st.median(pool):+13.2f} [{lo:+8.2f},{hi:+8.2f}]{sig} {win:11.1f}%")
        print()
print("Positive median diff = adaptive better. * = 95% CI excludes zero.")
