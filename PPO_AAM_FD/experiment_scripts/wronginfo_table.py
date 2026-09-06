"""Wrong-information control, paired per environment.

The decisive comparison is TRUE vs SHUFFLED, not each against frozen. Both arms
carry a context of the same distribution and magnitude on the same mixed-payload
plant; only the pairing between the number and the environment differs. If the
claim is that PAYLOAD INFORMATION does the work, true must beat shuffled.
"""
import json, os, random, statistics as st

B = 20000
ROOT = "outputs/final"


def row(p):
    d = json.load(open(os.path.join(ROOT, p)))["per_mesh"]
    return list(d.values())[0]


def cmp(a, b, key, unit=1000.0, higher_better=False):
    x = [v * unit for v in a[key]]
    y = [v * unit for v in b[key]]
    n = min(len(x), len(y))
    d = [(y[i] - x[i]) if higher_better else (x[i] - y[i]) for i in range(n)]
    random.seed(0)
    bs = sorted(st.median([d[random.randrange(n)] for _ in range(n)])
                for _ in range(B))
    lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B)]
    win = 100.0 * sum(1 for v in d if v > 0) / n
    return st.median(x), st.median(y), st.median(d), lo, hi, win


z, lv, sh = row("wronginfo_zero.json"), row("wronginfo_live.json"), row("wronginfo_shuffle.json")
print("Mixed payload 1.0-5.0x resampled per episode, seed 17 checkpoint, mesh 3141")
print("Positive median diff = SECOND arm better.\n")
for key, name, hb in (("per_env_cut_error_m", "cut (mm)", False),
                      ("per_env_lateral_m", "lateral (mm)", False),
                      ("per_env_contact_fraction", "contact", True)):
    unit = 1.0 if hb else 1000.0
    print(f"--- {name} ---")
    for an, a, bn, b in (("frozen", z, "true", lv),
                         ("frozen", z, "shuffled", sh),
                         ("shuffled", sh, "true", lv)):
        ma, mb, md, lo, hi, win = cmp(a, b, key, unit, hb)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {an:>9} {ma:8.2f}  ->  {bn:>9} {mb:8.2f}   "
              f"diff {md:+8.2f} [{lo:+8.2f},{hi:+8.2f}]{sig}  better in {win:.1f}%")
    print()
