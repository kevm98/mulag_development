"""Five-seed table for method R1, plus the median-seed verdict.

Reuses verdict.py's statistics exactly: paired over the same environments and
meshes, median with a bootstrap CI on the paired difference, and the
pre-registered bar that BOTH metrics must improve with the CI excluding zero.

Usage:
  seed_table.py --base g_base --arms 17=rebal17_history 101=rebal101_history ...
"""
import json, os, sys, math, random, statistics as st

B = 20000
MESHES = ("1337", "17", "4242")
D = "outputs/rif_delay"
KEYS = (("per_env_cut_error_m", "cut"), ("per_env_lateral_m", "lateral"))


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


def compare(a, b, seed=0):
    """a = reference, b = arm. Positive % => b is better (lower error)."""
    random.seed(seed)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    diffs = [x - y for x, y in zip(a, b)]
    med = st.median(diffs)
    boot = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)])
                  for _ in range(B))
    lo, hi = boot[int(.025 * B)], boot[int(.975 * B)]
    lr = [math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0]
    pct = (math.exp(st.median(lr)) - 1) * 100
    return dict(n=n, med_a=st.median(a), med_b=st.median(b), diff=med,
                lo=lo, hi=hi, pct=pct, sig=(lo > 0 or hi < 0))


def arm_verdict(base_tag, arm_tag):
    out = {}
    for key, name in KEYS:
        a, b = pool(base_tag, key), pool(arm_tag, key)
        if a is None or b is None:
            return None
        out[name] = compare(a, b)
    out["pass"] = all(out[n]["pct"] > 0 and out[n]["sig"] for _, n in KEYS)
    return out


def main():
    args = sys.argv[1:]
    base = args[args.index("--base") + 1] if "--base" in args else "g_base"
    arms = []
    if "--arms" in args:
        for tok in args[args.index("--arms") + 1:]:
            if tok.startswith("--"):
                break
            arms.append(tuple(tok.split("=", 1)))

    print(f"Method R1 five-seed table. Base = {base}. "
          f"n=528 paired envs, meshes {', '.join(MESHES)}.")
    print("* = bootstrap CI on the paired difference excludes zero. "
          "PASS requires BOTH metrics improved and significant.\n")
    hdr = (f"{'seed':>6} | {'cut base':>9} {'cut arm':>9} {'cut %':>9} "
           f"| {'lat base':>9} {'lat arm':>9} {'lat %':>9} | verdict")
    print(hdr); print("-" * len(hdr))

    results, passes = {}, []
    for seed, tag in arms:
        v = arm_verdict(base, tag)
        if v is None:
            print(f"{seed:>6} | {'MISSING reports for ' + tag:>60}")
            continue
        c, l = v["cut"], v["lateral"]
        verdict = "PASS" if v["pass"] else (
            "both worse" if (c["pct"] < 0 and l["pct"] < 0) else "TRADE-OFF")
        print(f"{seed:>6} | {c['med_a']:9.3f} {c['med_b']:9.3f} "
              f"{c['pct']:+8.1f}%{'*' if c['sig'] else ' '}"
              f"| {l['med_a']:9.3f} {l['med_b']:9.3f} "
              f"{l['pct']:+8.1f}%{'*' if l['sig'] else ' '}| {verdict}")
        results[seed] = v
        passes.append(v["pass"])

    print()
    for seed, v in results.items():
        for _, name in KEYS:
            r = v[name]
            print(f"  seed {seed:>3} {name:>7}: paired {r['diff']:+7.3f} mm "
                  f"[{r['lo']:+7.3f}, {r['hi']:+7.3f}]")

    n_pass = sum(passes)
    print(f"\n{n_pass} of {len(passes)} seeds pass both metrics.")
    if len(passes) == 5:
        ok = n_pass >= 3
        print(f"MEDIAN SEED {'PASSES' if ok else 'FAILS'} "
              f"-- the pre-registered bar is {'MET' if ok else 'NOT MET'}.")
        for _, name in KEYS:
            meds = sorted(v[name]["pct"] for v in results.values())
            print(f"  median across seeds, {name}: {meds[len(meds)//2]:+.1f}%  "
                  f"(all: {', '.join(f'{x:+.1f}' for x in meds)})")
    else:
        print(f"INCOMPLETE: {len(passes)} of 5 seeds scored. "
              "No median verdict until all five are in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
