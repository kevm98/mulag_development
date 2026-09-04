"""Recompute the cutter-mass dose tables from the frozen reports."""
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RIF = ROOT / "outputs/rif_delay"
B = 20000
DOSES = ["1_0", "1_5", "2_0", "2_5", "3_0"]


def meshes_for(tag):
    return sorted({f.name.split("_m")[-1][:-5]
                   for f in RIF.glob(f"{tag}_m*.json")})


def pool(tag, key):
    v = []
    for m in meshes_for(tag):
        f = RIF / f"{tag}_m{m}.json"
        if not f.exists():
            return None
        d = json.load(open(f))
        for _, r in d["per_mesh"].items():
            v += [x * 1000.0 for x in r[key]]
    return v


def paired(base, adapt, seed=0):
    """Positive => adaptive has the lower error."""
    random.seed(seed)
    n = min(len(base), len(adapt))
    d = [base[i] - adapt[i] for i in range(n)]
    draws = sorted(st.median([d[random.randrange(n)] for _ in range(n)])
                   for _ in range(B))
    return st.median(d), draws[int(.025 * B)], draws[int(.975 * B)], n


def diff_of_gains(b0, a0, b1, a1, seed=0):
    """Bootstrap the DIFFERENCE of two paired median gains."""
    random.seed(seed)
    n0 = min(len(b0), len(a0))
    n1 = min(len(b1), len(a1))
    d0 = [b0[i] - a0[i] for i in range(n0)]
    d1 = [b1[i] - a1[i] for i in range(n1)]
    point = st.median(d1) - st.median(d0)
    draws = sorted(
        st.median([d1[random.randrange(n1)] for _ in range(n1)])
        - st.median([d0[random.randrange(n0)] for _ in range(n0)])
        for _ in range(B))
    return point, draws[int(.025 * B)], draws[int(.975 * B)]


for key, label in [("per_env_lateral_m", "LATERAL"),
                   ("per_env_cut_error_m", "CUT HEIGHT")]:
    print(f"\n================ {label} ================")
    ref = None
    rows = []
    for dose in DOSES:
        b = pool(f"dose_base_{dose}", key)
        a = pool(f"dose_adapt_{dose}", key)
        if b is None or a is None:
            print(f"  dose {dose}: MISSING")
            continue
        g, lo, hi, n = paired(b, a)
        if dose == "1_0":
            ref = (b, a, g)
            print(f"  dose {dose.replace('_','.')}  total gain {g:8.3f}  "
                  f"[{lo:7.3f},{hi:7.3f}]  n={n}   (sham reference)")
            continue
        ep, elo, ehi = diff_of_gains(ref[0], ref[1], b, a)
        print(f"  dose {dose.replace('_','.')}  total gain {g:8.3f}  "
              f"[{lo:7.3f},{hi:7.3f}]  n={n}")
        print(f"           excess over 1.0: bootstrap {ep:8.3f} "
              f"[{elo:7.3f},{ehi:7.3f}]   naive difference of medians "
              f"{g - ref[2]:8.3f}")
