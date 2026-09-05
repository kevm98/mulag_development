"""The oracle table: does a fixed task-space offset Pareto-improve this stack?

Design, grid, success rule and clearance guard are fixed in
reports/PREREG_ORACLE_PHI_2026-09-06.md and are NOT re-decided here.

The statistic is triplet.py's, unchanged: paired per-environment differences
against the 0:0 baseline measured in the SAME process, median, percentile
bootstrap B = 20000, percentage as the median log-ratio. Non-inferiority bounds
are the A/A floors of PREREG_R2_2026-09-01.md (cut 10%, lateral 1.5%).

What is new here is the clearance guard. cut_height_error is relu(h - reference)
and exactly zero in contact, so an offset that drives the cutter into the ground
scores a perfect cut height by destroying clearance. An offset that wins on both
error channels but drops worst-family ground contact below the E5 criterion of
0.700, or raises the safety-violation rate above the baseline's, is reported as
a metric artefact and does not count as a pass.

Usage:  oracle_phi.py [--dir outputs/oracle_phi] [--mesh 2718]
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
CONTACT_FLOOR = 0.700
BASELINE = "0.0:0.0"


def compare(ref, arm):
    """ref = baseline, arm = offset. Positive pct => the offset is better."""
    random.seed(0)
    n = min(len(ref), len(arm))
    ref, arm = ref[:n], arm[:n]
    diffs = [x - y for x, y in zip(ref, arm)]
    boot = sorted(
        st.median([diffs[random.randrange(n)] for _ in range(n)])
        for _ in range(B)
    )
    lo, hi = boot[int(0.025 * B)], boot[int(0.975 * B)]
    lr = [math.log(x / y) for x, y in zip(ref, arm) if x > 0 and y > 0]
    pct = (math.exp(st.median(lr)) - 1) * 100 if lr else float("nan")
    return dict(n=n, pct=pct, lo=lo, hi=hi, sig=(lo > 0 or hi < 0),
                med_ref=st.median(ref), med_arm=st.median(arm))


def label(res):
    """The pre-registered success criterion, identical to triplet.py's."""
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


def guard(arm, base):
    """The clearance guard. Returns (ok, reason)."""
    contact = arm["worst_family_ground_contact_fraction"]
    if contact < CONTACT_FLOOR:
        return False, f"contact {contact:.3f} < {CONTACT_FLOOR}"
    if arm["safety_violation_rate"] > base["safety_violation_rate"]:
        return False, (f"safety {arm['safety_violation_rate']:.3f} > "
                       f"{base['safety_violation_rate']:.3f}")
    return True, ""


def sweep(path):
    """Pull the paired offset sweep out of one report."""
    doc = json.load(open(path))
    for mesh in doc.get("per_mesh", {}).values():
        if "paired_task_bias" in mesh:
            return mesh["paired_task_bias"]
    return None


def table(path, payload):
    arms = sweep(path)
    if arms is None:
        print(f"  {payload}: no paired_task_bias block in {path}")
        return None
    if BASELINE not in arms:
        print(f"  {payload}: no {BASELINE} baseline in {path}")
        return None
    base = arms[BASELINE]
    b_cut = [v * 1000.0 for v in base["per_env_cut_error_m"]]
    b_lat = [v * 1000.0 for v in base["per_env_lateral_m"]]

    print(f"\n{'=' * 92}")
    print(f"payload {payload}x   baseline: cut {st.median(b_cut):.3f} mm  "
          f"lateral {st.median(b_lat):.3f} mm  "
          f"contact {base['worst_family_ground_contact_fraction']:.3f}  "
          f"safety {base['safety_violation_rate']:.3f}")
    print('=' * 92)
    # Absolute millimetres alongside the percentages. At light payload the
    # baseline cut error is ~1.6 mm, where the pre-registered 10% bound is
    # 0.16 mm, well under the instrument's own floor, and a percentage on that
    # denominator says more about the denominator than about the offset.
    print(f"{'b_Y:b_Z':>12}  {'cut':>10} {'d_mm':>7}  {'lateral':>10} "
          f"{'d_mm':>7}  {'contact':>8}  {'safety':>7}   outcome")

    winners = []
    for key, arm in arms.items():
        if key == BASELINE:
            continue
        res = {
            "cut": compare(b_cut, [v * 1000.0 for v in arm["per_env_cut_error_m"]]),
            "lateral": compare(b_lat, [v * 1000.0 for v in arm["per_env_lateral_m"]]),
        }
        lab, ok = label(res)
        guard_ok, why = guard(arm, base)
        if ok and not guard_ok:
            lab = f"ARTEFACT: {lab}, {why}"
            ok = False
        elif ok:
            winners.append((key, res))
        cell = lambda r: f"{r['pct']:+9.1f}%{'*' if r['sig'] else ' '}"
        dmm = lambda r: f"{r['med_arm'] - r['med_ref']:+7.2f}"
        print(f"{key:>12}  {cell(res['cut'])}{dmm(res['cut'])}  "
              f"{cell(res['lateral'])}{dmm(res['lateral'])}  "
              f"{arm['worst_family_ground_contact_fraction']:8.3f}  "
              f"{arm['safety_violation_rate']:7.3f}   {lab}")

    if winners:
        best = max(winners, key=lambda w: min(w[1]["cut"]["pct"],
                                              w[1]["lateral"]["pct"]))
        print(f"\n  PASS at {payload}x: best phi = {best[0]} "
              f"(cut {best[1]['cut']['pct']:+.1f}%, "
              f"lateral {best[1]['lateral']['pct']:+.1f}%)")
        return best[0]
    print(f"\n  FAIL at {payload}x: no offset on this grid meets the rule.")
    return None


if __name__ == "__main__":
    root = (sys.argv[sys.argv.index("--dir") + 1]
            if "--dir" in sys.argv else "outputs/oracle_phi")
    mesh = (sys.argv[sys.argv.index("--mesh") + 1]
            if "--mesh" in sys.argv else "2718")

    print(f"Oracle probe, mesh {mesh}. Paired per-environment differences "
          f"against the 0:0 arm\nmeasured in the same process. Median, "
          f"bootstrap CI B={B}. * = CI excludes zero.")
    print(f"Non-inferiority: cut {BOUND['cut']}%, lateral {BOUND['lateral']}%. "
          f"Clearance guard: worst-family\ncontact >= {CONTACT_FLOOR} and "
          f"safety violation not above baseline.")

    files = sorted(glob.glob(os.path.join(root, f"stage1_m{mesh}_p*.json")),
                   key=lambda f: float(f.rsplit("_p", 1)[1][:-5]))
    if not files:
        raise SystemExit(f"no stage1_m{mesh}_p*.json under {root}")

    verdicts = {}
    for path in files:
        payload = path.rsplit("_p", 1)[1][:-5]
        verdicts[payload] = table(path, payload)

    print(f"\n{'=' * 92}\nVERDICT\n{'=' * 92}")
    for payload, best in verdicts.items():
        print(f"  {payload}x  " + (f"pass, phi = {best}" if best else "FAIL"))
    passed = [p for p, b in verdicts.items() if b]
    if len(passed) == len(verdicts):
        print("\n  Every payload passes. The target exists; build the estimator.")
    elif passed:
        print(f"\n  PARTIAL: {len(passed)}/{len(verdicts)} payloads pass "
              f"({', '.join(passed)}x). Report the boundary, do not round up.")
    else:
        print("\n  No payload passes. A two-parameter task-space offset cannot "
              "Pareto-improve\n  this stack. Do NOT build the estimator; the "
              "parameterisation is what must change.")
