#!/usr/bin/env python3
"""Actuation-chain latency, measured separately on each of the three stacks.

The open-loop measurement in `transport_delay.py` uses onset detection on
square-wave edges, which cleanly separates dead time from rise time. A
closed-loop command has no such edges -- it varies continuously -- so the only
usable estimator here is a cross-correlation lag, which conflates the two.

That makes the numbers below valid as a COMPARISON across stacks, where the
bias is identical for all three, and NOT directly comparable to the open-loop
dead times. The two must not be quoted as if they measured the same quantity.

Chain, per cylinder:
    valve command -> differential pressure   (hydraulic)
    differential pressure -> cylinder velocity (hydro-mechanical)
    cylinder extension -> joint angle          (mechanical linkage, static)

The last link is the CONTROL: the linkage is geometric, so its lag must come
out at zero on every stack. If it does not, the estimator is manufacturing lag
and nothing else in the table can be trusted.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONTROL_HZ = 60.0
STACKS = [("ppo_rom", "PPO-ROM"), ("ppo_id_fd", "PPO-ID-FD"),
          ("ppo_aam_id_fd", "PPO-(AAM)-ID-FD")]


def xcorr_lag(drive, resp, max_lag):
    """Lag maximising correlation of the differenced signals, and that value."""
    d = np.diff(drive, axis=0)
    r = np.diff(resp, axis=0)
    d = d - d.mean(0, keepdims=True)
    r = r - r.mean(0, keepdims=True)
    best, best_c = 0, -2.0
    for L in range(0, max_lag + 1):
        a = d[:len(d) - L] if L else d
        b = r[L:]
        n = min(len(a), len(b))
        if n < 64:
            break
        num = float((a[:n] * b[:n]).sum())
        den = float(np.linalg.norm(a[:n]) * np.linalg.norm(b[:n])) + 1e-12
        c = num / den
        if c > best_c:
            best, best_c = L, c
    return best, best_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mesh", default="1337")
    ap.add_argument("--max-lag", type=int, default=40)
    ap.add_argument("--min-corr", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    links = [("command $\\to$ pressure", "valve_cmd", "chain_dP", None),
             ("pressure $\\to$ cyl. velocity", "chain_dP", "chain_cyl_vel", None),
             ("command $\\to$ cyl. velocity", "valve_cmd", "chain_cyl_vel", None),
             # The control uses VELOCITIES, not positions. Extension and joint
             # angle are slowly varying, their differenced signals carry almost
             # no high-frequency content, and the correlation surface is flat
             # enough that argmax lands on the search boundary rather than on a
             # peak -- which is exactly what a first attempt produced.
             ("cyl. velocity $\\to$ joint velocity", "chain_cyl_vel",
              "chain_joint_vel", "map")]

    report, table = {"control_hz": CONTROL_HZ, "stacks": {}}, {}
    for key, label in STACKS:
        p = a.root / f"{key}_m{a.mesh}.npz"
        if not p.is_file():
            print(f"  missing {p}")
            continue
        d = np.load(p)
        # pair each cylinder with the joint it drives, resolved at zero lag
        ext, jp = d["chain_cyl_vel"], d["chain_joint_vel"]
        pair = []
        for i in range(ext.shape[2]):
            de = np.diff(ext[:, :, i], axis=0).ravel()
            best, bc = 0, -2.0
            for j in range(jp.shape[2]):
                dj = np.diff(jp[:, :, j], axis=0).ravel()
                c = abs(float(np.dot(de - de.mean(), dj - dj.mean()) /
                              (np.linalg.norm(de - de.mean()) *
                               np.linalg.norm(dj - dj.mean()) + 1e-12)))
                if c > bc:
                    best, bc = j, c
            pair.append(best)

        row = {}
        for name, dk, rk, mode in links:
            lags = []
            for i in range(d[dk].shape[2]):
                j = pair[i] if mode == "map" else min(i, d[rk].shape[2] - 1)
                for e in range(d[dk].shape[1]):
                    L, c = xcorr_lag(d[dk][:, e, i], d[rk][:, e, j], a.max_lag)
                    if c >= a.min_corr:
                        lags.append(L)
            row[name] = (float(np.median(lags)) if lags else float("nan"),
                         len(lags))
        table[label] = row
        report["stacks"][label] = {k: {"median_steps": v[0],
                                       "median_ms": 1000 * v[0] / CONTROL_HZ,
                                       "n": v[1]} for k, v in row.items()}

    names = [l[0] for l in links]
    print(f"\n  cross-correlation lag, median over cylinders and environments")
    print(f"  {'link':<30}" + "".join(f"{lab:>18}" for _, lab in STACKS))
    for n in names:
        cells = []
        for _, lab in STACKS:
            v = table.get(lab, {}).get(n, (float('nan'), 0))[0]
            cells.append(f"{1000*v/CONTROL_HZ:>15.1f} ms" if v == v else f"{'-':>18}")
        print(f"  {n.replace(chr(92)+'to', '->').replace('$',''):<30}" + "".join(cells))

    ctrl = names[-1]
    worst = max(abs(table[lab][ctrl][0]) for _, lab in STACKS if lab in table)
    ok = worst <= 1.0
    print(f"\n  CONTROL (static linkage) worst median {worst:.1f} steps  "
          f"{'OK' if ok else 'FAILED -- estimator is inventing lag'}")
    report["control_passed"] = bool(ok)
    report["control_worst_steps"] = float(worst)
    print("\n  NOTE: cross-correlation conflates dead time with rise time. These "
          "\n  values compare stacks; they are not the open-loop dead times.")
    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
