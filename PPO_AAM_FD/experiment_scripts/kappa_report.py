"""kappa -> performance, and kappa_crit per stack.

Thresholds are E5's, declared before any curve was seen
(experiments/E5/README.md):

  functional breakdown : ground contact < 0.700  OR  safety termination > 0.080
  degradation onset    : median lateral >= 1.5x its own kappa=1.0 value
                         OR ground contact falls >= 5 percentage points

kappa_crit is the LARGEST kappa at which the stack has broken down -- i.e. the
authority level it can no longer survive. Reported per stack, plus the interval
between the last surviving and first failing rung.
"""
import json, os, sys, statistics as st

D = "outputs/kappa"
STACKS = [("R", "ROM"), ("IF", "ID->FD"), ("AFf", "AF frozen"), ("AFa", "AF adaptive")]
CONTACT_GATE, SAFETY_GATE = 0.700, 0.080


def read(stack, k, mesh):
    f = f"{D}/{stack}_k{k}_m{mesh}.json"
    if not os.path.exists(f):
        return None
    r = list(json.load(open(f))["per_mesh"].values())[0]
    return dict(
        k=float(k),
        lateral=st.median([x * 1000 for x in r["per_env_lateral_m"]]),
        cut=st.median([x * 1000 for x in r["per_env_cut_error_m"]]),
        contact=float(r["worst_family_ground_contact_fraction"]),
        safety=float(r.get("safety_termination_rate", 0.0)),
    )


def broken(row):
    return row["contact"] < CONTACT_GATE or row["safety"] > SAFETY_GATE


def main():
    mesh = sys.argv[sys.argv.index("--mesh") + 1] if "--mesh" in sys.argv else "1337"
    stage0 = "--stage0" in sys.argv
    ks = ["1.0", "0.01"] if stage0 else \
         [x for x in ("1.0", "0.8", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1")]

    print(f"\nmesh {mesh}   breakdown: contact < {CONTACT_GATE} or safety > {SAFETY_GATE}")
    print(f"{'stack':12} {'kappa':>6} {'lateral':>9} {'cut':>8} {'contact':>8} {'safety':>7}  state")
    crit = {}
    for s, label in STACKS:
        rows = [r for r in (read(s, k, mesh) for k in ks) if r]
        if not rows:
            print(f"{label:12} (no data)"); continue
        for r in rows:
            state = "BROKEN" if broken(r) else "ok"
            print(f"{label:12} {r['k']:6.2f} {r['lateral']:9.2f} {r['cut']:8.3f} "
                  f"{r['contact']:8.4f} {r['safety']:7.3f}  {state}")
        if stage0:
            hi = next((r for r in rows if r["k"] == 1.0), None)
            lo = next((r for r in rows if r["k"] == 0.01), None)
            if hi and lo:
                ratio = lo["lateral"] / hi["lateral"] if hi["lateral"] > 0 else float("nan")
                # Health at kappa=1.0 is REQUIRED: a stack already broken at
                # full authority has no envelope to measure, and "broken at
                # kappa=0.01" is then not evidence the stress arrived.
                if broken(hi):
                    verdict = "*** EXCLUDED: already broken at kappa=1.0 ***"
                elif ratio > 1.5 or broken(lo):
                    verdict = "PASS"
                else:
                    verdict = "*** FAIL: stress did not arrive ***"
                print(f"{'':12} gate: lateral x{ratio:.2f} at kappa=0.01   {verdict}")
        else:
            ok = [r for r in rows if not broken(r)]
            bad = [r for r in rows if broken(r)]
            if bad:
                crit[label] = (max(r["k"] for r in bad),
                               min((r["k"] for r in ok if r["k"] > max(r2["k"] for r2 in bad)),
                                   default=float("nan")))
        print()

    if crit:
        print("kappa_crit  (largest kappa at which the stack has broken down)")
        for label, (kc, last_ok) in crit.items():
            print(f"  {label:12} kappa_crit = {kc:.2f}   last surviving rung {last_ok:.2f}")
        if "AF frozen" in crit and "AF adaptive" in crit:
            f, a = crit["AF frozen"][0], crit["AF adaptive"][0]
            if a < f:
                print(f"\n  AF adaptive survives to LOWER authority than AF frozen "
                      f"({a:.2f} vs {f:.2f}) -- adaptation extends the envelope.")
            else:
                print(f"\n  AF adaptive does NOT survive lower authority than frozen "
                      f"({a:.2f} vs {f:.2f}).")


if __name__ == "__main__":
    main()
