"""The payload-conditioning table.

Design, arms and success rule are fixed in
reports/PREREG_PAYLOAD_CONDITIONING_2026-09-06.md and are NOT re-decided here.

Statistic is triplet.py's, unchanged: paired per-environment differences,
median, percentile bootstrap B = 20000, percentage as the median log-ratio,
non-inferiority bounds cut 10% and lateral 1.5%. Arms are separate processes
with common random numbers, paired by environment index (invariant 3).

The comparisons, in the order the pre-registration ranks them:

  blind    -> adaptive    PRIMARY. Isolates payload information: both arms had
                          the same extra training on the same wide-payload
                          plant, and both started as literally the same policy
                          because the base's context weights are exactly zero.
  adaptive -> wrong       FALSIFICATION at fixed payload. Same weights, told
                          it carries nominal while the plant gives it 3x or 5x.
                          A PERMUTATION would be degenerate here: at fixed
                          payload every environment has the same mass_scale, so
                          shuffling the rows is the identity. The permuted
                          control is run separately on the mixed range, where
                          environments genuinely differ.
  base     -> blind       CONTEXT. How much of the deficit is recovered by
                          training on the range with no plant information.
  base     -> adaptive    CONFOUNDED, reported and labelled as such. It mixes
                          training-on-the-range with payload information and is
                          NOT the headline.

Secondary outcomes are the root-cause quantities, declared before any arm ran:
contact, hard impacts, p95 tool acceleration, safety violations.

Usage:  payload_table.py [--dir outputs/payload_cond] [--mesh 2718]
"""
import glob
import json
import math
import os
import random
import statistics as st
import sys

B = 20000
BOUND = {"cut": 10.0, "lateral": 1.5}
ARMS = ("base", "blind", "adaptive", "adaptive_wrong")


def load(root, arm, payload, mesh):
    path = os.path.join(root, f"{arm}_p{payload}_m{mesh}.json")
    if not os.path.exists(path):
        return None
    doc = json.load(open(path))
    for r in doc.get("per_mesh", {}).values():
        return r
    return None


def compare(ref, arm):
    """ref = reference arm, arm = test arm. Positive pct => test is better."""
    random.seed(0)
    n = min(len(ref), len(arm))
    ref, arm = ref[:n], arm[:n]
    diffs = [x - y for x, y in zip(ref, arm)]
    boot = sorted(st.median([diffs[random.randrange(n)] for _ in range(n)])
                  for _ in range(B))
    lo, hi = boot[int(0.025 * B)], boot[int(0.975 * B)]
    lr = [math.log(x / y) for x, y in zip(ref, arm) if x > 0 and y > 0]
    pct = (math.exp(st.median(lr)) - 1) * 100 if lr else float("nan")
    return dict(n=n, pct=pct, lo=lo, hi=hi, sig=(lo > 0 or hi < 0),
                med_ref=st.median(ref), med_arm=st.median(arm))


def label(res):
    wins = [k for k, r in res.items() if r["pct"] > 0 and r["sig"]]
    if not wins:
        if all(r["pct"] < 0 and r["sig"] for r in res.values()):
            return "both worse", False
        return "no improvement", False
    if len(wins) == 2:
        return "joint improvement", True
    other = "lateral" if wins[0] == "cut" else "cut"
    if res[other]["pct"] >= -BOUND[other]:
        return f"Pareto ({other} maintained)", True
    return f"trade-off ({other} worse)", False


def pair(ref_row, arm_row):
    out = {}
    for key, name in (("per_env_cut_error_m", "cut"),
                      ("per_env_lateral_m", "lateral")):
        out[name] = compare([v * 1000.0 for v in ref_row[key]],
                            [v * 1000.0 for v in arm_row[key]])
    return out


def secondary(row):
    return (st.mean(row["per_env_contact_fraction"]),
            row["hard_impact_step_fraction"],
            row["p95_tool_acceleration_mps2"],
            row["safety_violation_rate"])


def run(root, mesh, payload):
    rows = {a: load(root, a, payload, mesh) for a in ARMS}
    missing = [a for a, r in rows.items() if r is None]
    if missing:
        print(f"\npayload {payload}x: MISSING {missing}")
        return None

    print(f"\n{'=' * 100}")
    print(f"payload {payload}x, mesh {mesh}")
    print('=' * 100)
    print(f"{'arm':>17}  {'cut mm':>9} {'lat mm':>9}  {'contact':>8} "
          f"{'impact':>7} {'accel':>7} {'safety':>7}")
    for a in ARMS:
        r = rows[a]
        c, i, ac, s = secondary(r)
        print(f"{a:>17}  "
              f"{st.median([v * 1000 for v in r['per_env_cut_error_m']]):9.2f} "
              f"{st.median([v * 1000 for v in r['per_env_lateral_m']]):9.2f}  "
              f"{c:8.3f} {i:7.3f} {ac:7.1f} {s:7.3f}")

    cell = lambda r: f"{r['pct']:+9.1f}%{'*' if r['sig'] else ' '}"
    print(f"\n{'comparison':>34}  {'cut':>10}  {'lateral':>10}   outcome")
    verdict = None
    for ref, arm, tag, is_primary in (
        ("blind", "adaptive", "PRIMARY   blind -> adaptive", True),
        ("adaptive", "adaptive_wrong", "FALSIFY   adaptive -> wrong estimate", False),
        ("base", "blind", "CONTEXT   base -> blind", False),
        ("base", "adaptive", "CONFOUND  base -> adaptive", False),
    ):
        res = pair(rows[ref], rows[arm])
        lab, ok = label(res)
        print(f"{tag:>34}  {cell(res['cut'])}  {cell(res['lateral'])}   {lab}")
        if is_primary:
            verdict = (ok, res)
    return verdict


if __name__ == "__main__":
    root = (sys.argv[sys.argv.index("--dir") + 1]
            if "--dir" in sys.argv else "outputs/payload_cond")
    mesh = (sys.argv[sys.argv.index("--mesh") + 1]
            if "--mesh" in sys.argv else "2718")
    found = sorted({os.path.basename(f).split("_p")[1].split("_m")[0]
                    for f in glob.glob(os.path.join(root, f"*_p*_m{mesh}.json"))},
                   key=float)
    if not found:
        raise SystemExit(f"no arm reports under {root} for mesh {mesh}")
    print(f"Payload conditioning, mesh {mesh}. Paired per-environment "
          f"differences, B={B}.\n* = CI excludes zero. Bounds: cut "
          f"{BOUND['cut']}%, lateral {BOUND['lateral']}%.")
    verdicts = {}
    for payload in found:
        verdicts[payload] = run(root, mesh, payload)

    print(f"\n{'=' * 100}\nVERDICT (primary: blind -> adaptive)\n{'=' * 100}")
    for payload, v in verdicts.items():
        if v is None:
            print(f"  {payload}x  incomplete")
        else:
            ok, res = v
            print(f"  {payload}x  {'PASS' if ok else 'fail'}  "
                  f"cut {res['cut']['pct']:+.1f}%  "
                  f"lateral {res['lateral']['pct']:+.1f}%")
    done = [v for v in verdicts.values() if v is not None]
    if done and all(v[0] for v in done):
        print("\n  Payload information helps at every scored payload. Check the "
              "falsification\n  row before claiming it: a shuffled context that "
              "matches the live one voids this.")
