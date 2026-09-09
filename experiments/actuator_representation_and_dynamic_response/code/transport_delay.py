#!/usr/bin/env python3
"""Transport latency along the actuation chain, measured from recorded traces.

This measures TIMING, not valve parameters. An earlier attempt to identify the
valve's dynamic model from the MAT and ARX datasets returned null in both, but
that is a parameter-fitting result and says nothing about whether the plant's
command-to-response latency is measurable. It is: the open-loop collection
drives zero-mean square waves with known edge times, which is what one would
design for a step-response latency measurement.

Chain, per cylinder:

    valve command  --(a)-->  differential pressure  --(b)-->  cylinder velocity
                                                          --(c)-->  joint velocity

(a) and (b) are physical transport: valve travel, fluid compliance, line
dynamics. (c) is the CONTROL. Cylinder extension drives joint angle through a
static geometric linkage, so the cylinder-to-joint lag must come out at or near
zero. If it does not, the estimator is manufacturing lag and none of the other
numbers can be trusted -- a static kinematic mapping must never be reported as
a temporal delay.

Method: ONSET detection, not cross-correlation. A correlation peak conflates
dead time with rise time -- a larger command crosses any detection threshold
sooner, which shows up as a shorter "delay" at higher amplitude even when the
transport lag is unchanged. Instead, each command edge is located, a noise floor
is measured on the samples immediately BEFORE it, and the dead time is the first
sample after the edge at which the response leaves that floor by a fixed number
of standard deviations. That quantity is amplitude-invariant when the underlying
delay is.

The cylinder-to-joint linkage is resolved from the data rather than assumed:
there are five joints and four cylinders and the indices are not aligned, so
each cylinder is paired with the joint whose motion best matches it at zero lag.

The CONTROL compares command->cylinder_extension against command->joint_position
using the same estimator and the same driving edges. Both are onsets measured
from the same square wave, so they are directly comparable, and their DIFFERENCE
is what the static linkage contributes. It must be near zero. Measuring the
linkage by feeding extension itself into an onset detector would be invalid --
extension is smooth and has no edges to detect.

Latency is reported per amplitude tier, because the tiers differ in saturation
and a saturated actuator responds late for a reason that is not transport.

CPU only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CONTROL_HZ = 60.0


def tiers_of(meta: dict) -> dict[float, tuple[int, int]]:
    out: dict[float, list[int]] = {}
    for s in meta["segment_ids"]:
        if s.get("kind") != "probe":
            continue
        m = re.search(r"_a([0-9.]+)_", s["name"])
        if not m:
            continue
        amp = float(m.group(1))
        lo, hi = out.setdefault(amp, [s["start"], s["end"]])
        out[amp] = [min(lo, s["start"]), max(hi, s["end"])]
    return {a: (lo, hi) for a, (lo, hi) in sorted(out.items())}


def edges(cmd: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    """Indices of the large command steps that drive a measurable response."""
    d = np.abs(np.diff(cmd))
    big = d > max(np.quantile(d, quantile), 1e-9)
    idx = np.flatnonzero(big) + 1
    if idx.size == 0:
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] > 8:      # one onset per edge
            keep.append(i)
    return np.array(keep)


def onset_lag(cmd, resp, max_lag, pre=12, k=4.0) -> list[float]:
    """Dead time per command edge: first sample leaving the pre-edge noise floor."""
    out = []
    for e in edges(cmd):
        if e - pre < 0 or e + max_lag + 1 >= len(resp):
            continue
        base = resp[e - pre:e]
        mu, sd = base.mean(), base.std()
        if sd < 1e-12:
            sd = np.abs(resp[e:e + max_lag]).std()
            if sd < 1e-12:
                continue
        seg = np.abs(resp[e:e + max_lag + 1] - mu)
        hit = np.flatnonzero(seg > k * sd)
        if hit.size:
            out.append(float(hit[0]))
    return out


def joint_for_cylinder(cext: np.ndarray, jpos: np.ndarray) -> list[int]:
    """Pair each cylinder with the joint whose motion best matches it at lag 0."""
    dc, dj = np.diff(cext, axis=0), np.diff(jpos, axis=0)
    pairs = []
    for i in range(dc.shape[1]):
        a = dc[:, i] - dc[:, i].mean()
        best, bc = 0, -2.0
        for j in range(dj.shape[1]):
            b = dj[:, j] - dj[:, j].mean()
            c = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            if c > bc:
                best, bc = j, c
        pairs.append(best)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=24)
    ap.add_argument("--max-lag", type=int, default=40, help="control steps")
    ap.add_argument("--k-sigma", type=float, default=4.0,
                    help="onset threshold in pre-edge noise standard deviations")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    runs = sorted((a.raw_root / "runs").glob("[0-9]*"))[:a.runs]
    meta = json.loads((runs[0] / "metadata.json").read_text())
    tiers = tiers_of(meta)

    # link -> per tier -> list of lags in control steps
    acc: dict[str, dict[float, list[float]]] = {
        k: {t: [] for t in tiers} for k in
        ("command->dP", "dP->cyl_vel", "command->cyl_vel",
         "command->cyl_ext", "command->joint_pos")}

    mapping = None
    for rd in runs:
        d = np.load(rd / "trajectory.npz")
        n = int(meta["num_control_steps"])
        cmd = d["command_after_limits"][:n]
        dP = d["dP"][:n]
        cvel = d["cylinder_velocity"][:n]
        cext = d["cylinder_extension"][:n]
        jpos = d["joint_position"][:n]
        if mapping is None:
            mapping = joint_for_cylinder(cext, jpos)
            print(f"  cylinder -> joint linkage, resolved from data: {mapping}")
        for amp, (lo, hi) in tiers.items():
            sl = slice(lo, hi)
            for i in range(cmd.shape[1]):
                j = mapping[i]
                for name, drive, resp in (
                        ("command->dP", cmd[sl, i], dP[sl, i]),
                        ("dP->cyl_vel", dP[sl, i], cvel[sl, i]),
                        ("command->cyl_vel", cmd[sl, i], cvel[sl, i]),
                        ("command->cyl_ext", cmd[sl, i], cext[sl, i]),
                        ("command->joint_pos", cmd[sl, i], jpos[sl, j]),
                ):
                    acc[name][amp].extend(
                        onset_lag(drive, resp, a.max_lag, k=a.k_sigma))

    print(f"\n  {len(runs)} runs, {CONTROL_HZ:.0f} Hz control "
          f"({1000/CONTROL_HZ:.1f} ms per step)\n")
    print(f"  {'link':<20}{'amp':>7}{'median lag':>13}{'ms':>9}{'IQR (steps)':>16}{'n':>6}")
    report = {"control_hz": CONTROL_HZ, "runs": len(runs), "links": {}}
    for name, per_tier in acc.items():
        report["links"][name] = {}
        for amp, vals in per_tier.items():
            if len(vals) < 8:
                print(f"  {name:<20}{amp:>7.2f}{'too few responses':>13}")
                continue
            v = np.array(vals)
            med = float(np.median(v))
            q1, q3 = np.percentile(v, [25, 75])
            print(f"  {name:<20}{amp:>7.2f}{med:>13.1f}{1000*med/CONTROL_HZ:>9.1f}"
                  f"{f'{q1:.0f}-{q3:.0f}':>16}{len(v):>6}")
            report["links"][name][str(amp)] = {
                "median_steps": med, "median_ms": 1000 * med / CONTROL_HZ,
                "iqr_steps": [float(q1), float(q3)], "n": len(v)}
        print()

    ext = report["links"].get("command->cyl_ext", {})
    jnt = report["links"].get("command->joint_pos", {})
    diffs = {t: jnt[t]["median_steps"] - ext[t]["median_steps"]
             for t in ext if t in jnt}
    if diffs:
        worst = max(abs(v) for v in diffs.values())
        ok = worst <= 1.0
        print("  CONTROL: the cylinder->joint linkage is static, so the two "
              "onsets must coincide.")
        for t in sorted(diffs, key=float):
            print(f"    amp {float(t):.2f}: cyl_ext {ext[t]['median_steps']:.1f} "
                  f"vs joint {jnt[t]['median_steps']:.1f} steps  "
                  f"-> linkage adds {diffs[t]:+.1f}")
        print(f"  worst |difference| {worst:.1f} steps  "
              f"{'OK -- no lag manufactured' if ok else 'FAILED -- estimator is inventing lag'}")
        report["kinematic_control_passed"] = bool(ok)
        report["kinematic_control_worst_abs_diff_steps"] = float(worst)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
