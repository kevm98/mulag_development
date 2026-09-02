#!/usr/bin/env python3
"""Behavioural checks behind Table 3.x (Section 3.4.4, Validated Scope).

Reads the strict open-loop excitation export and compares it against the
parameters the collector configures in
``Randomization_open_loop/collect_open_loop.txt``.  Every number quoted in
Section~\\ref{sec:sys:simscape-validation} is printed here.

Usage:  python3 scripts/check_ch3_simscape_validation.py [path/to/export.parquet]
"""
import sys
import numpy as np
import pandas as pd

DEFAULT = (
    "/Volumes/DATA/rrlab_mulag/simscape_source/domain_randomization_collection/"
    "Domain_Randomization_Data_Collection/Randomization_open_loop/"
    "Excitation_Finetune_25/Excitation_FD_Strict.parquet"
)

# Declared by set_base_nominal_parameters() in collect_open_loop.txt.
P_RELIEF = 27.6e6           # prv_Pset, Pa
TAU_VALVE = 0.008           # tau_valve at nominal seal wear, s
X_MIN = np.array([0.00, 0.00, 0.00, 0.00])
X_MAX = np.array([1.85, 1.85, 2.25, 2.25])
DT = 0.01


def onset_and_rise(df):
    """Delay from command onset to motion onset, and time to 63 % of peak."""
    onset, rise = {i: [] for i in range(1, 5)}, {i: [] for i in range(1, 5)}
    for run in sorted(df.run_id.unique()):
        r = df[df.run_id == run].sort_values("t")
        for i in range(1, 5):
            u = np.abs(r[f"ValveCmd_{i}"].values)
            v = r[f"ExtensionVelocity_{i}"].values
            for k in range(30, len(u) - 100):
                if u[k] > 1e-6 and u[k - 30:k].max() < 1e-6:
                    base = v[k - 20:k].mean()
                    sd = v[k - 20:k].std() + 1e-6
                    fut = v[k:k + 100] - base
                    peak = np.abs(fut).max()
                    thr = max(0.05 * peak, 5 * sd)
                    onset[i].append(int(np.argmax(np.abs(fut) > thr)) * DT * 1e3)
                    rise[i].append(int(np.argmax(np.abs(fut) > 0.63 * peak)) * DT * 1e3)
    return onset, rise


def main(path):
    df = pd.read_parquet(path)
    d = df[df.valid == 1]
    print(f"rows {len(df)}  valid {len(d)} ({100 * len(d) / len(df):.1f} %)  "
          f"runs {len(df.run_id.unique())}  dt {DT} s")

    onset, rise = onset_and_rise(df)
    o = [np.median(onset[i]) for i in range(1, 5)]
    s = [np.median(rise[i]) for i in range(1, 5)]
    print(f"\nmotion onset (median per cylinder, ms): {o}  -> {min(o):.0f}-{max(o):.0f} ms")
    print(f"rise to 63 % of peak (ms):               {s}  -> {min(s):.0f}-{max(s):.0f} ms")
    print(f"nominal spool constant tau_valve = {TAU_VALVE * 1e3:.0f} ms")

    print("\nvelocity sign follows command (|u| > 0.5):")
    for thr in (0.01, 0.05):
        frac = []
        for i in range(1, 5):
            u = d[f"ValveCmd_{i}"].values
            v = d[f"ExtensionVelocity_{i}"].values
            m = (np.abs(u) > 0.5) & (np.abs(v) > thr)
            frac.append(100 * (np.sign(v[m]) == -np.sign(u[m])).mean())
        print(f"  |v| > {thr} m/s: " +
              "  ".join(f"C{i}={f:.1f} %" for i, f in enumerate(frac, 1)) +
              f"   min {min(frac):.1f} %")

    print("\nspeed against rising load, |u| > 0.5, |dP| terciles:")
    for i in range(1, 5):
        u = np.abs(d[f"ValveCmd_{i}"].values)
        v = np.abs(d[f"ExtensionVelocity_{i}"].values)
        a = np.abs(d[f"dP_{i}"].values)
        m = u > 0.5
        q = np.percentile(a[m], [33, 67])
        lo, mid, hi = m & (a <= q[0]), m & (a > q[0]) & (a <= q[1]), m & (a > q[1])
        print(f"  C{i}: {v[lo].mean():.4f} -> {v[mid].mean():.4f} -> {v[hi].mean():.4f} m/s "
              f"({np.median(a[lo]) / 1e6:.0f} -> {np.median(a[hi]) / 1e6:.0f} MPa)")

    print("\nstroke visited, as a fraction of the configured stroke:")
    cov = []
    for i in range(1, 5):
        x = df[f"Extension_{i}"].values
        cov.append(100 * (x.max() - x.min()) / (X_MAX[i - 1] - X_MIN[i - 1]))
        print(f"  C{i}: [{x.min():.3f}, {x.max():.3f}] m of "
              f"[{X_MIN[i - 1]:.2f}, {X_MAX[i - 1]:.2f}] m -> {cov[-1]:.0f} %")
    print(f"  range {min(cov):.0f}-{max(cov):.0f} %")

    print(f"\ndifferential pressure against the {P_RELIEF / 1e6:.1f} MPa relief setting:")
    ceil, above = [], []
    for i in range(1, 5):
        a = np.abs(d[f"dP_{i}"].values)
        ceil.append(a.max() / 1e6)
        above.append(100 * (a > 2 * P_RELIEF).mean())
        print(f"  C{i}: p50 {np.percentile(a, 50) / 1e6:5.1f}  p90 {np.percentile(a, 90) / 1e6:5.1f}  "
              f"max {ceil[-1]:5.1f} MPa   above 2x setting: {above[-1]:.1f} %")
    print(f"  ceiling {min(ceil):.0f}-{max(ceil):.0f} MPa, "
          f"{min(above):.0f}-{max(above):.0f} % of valid rows above twice the setting")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
