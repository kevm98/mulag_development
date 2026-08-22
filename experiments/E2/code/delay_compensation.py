#!/usr/bin/env python3
"""Has the policy compensated for the plant's dead time, and how?

The open-loop measurement establishes that a valve command needs about 117 ms
to reach differential pressure. A controller that ignored this would produce a
cutter path lagging its reference by roughly that interval. A controller that
compensates must issue its command EARLY, so that the response arrives on time.

Two lags are therefore measured on each stack, from the same closed-loop drive:

  ACHIEVED LAG    reference_y(t) against cutter_y(t)
                  what the task actually sees. Near zero means the dead time has
                  been absorbed.

  COMMAND LEAD    reference_y(t) against the valve command
                  negative lag means the command runs ahead of the reference,
                  which is the mechanism by which the dead time is absorbed.

The reference is sampled at the cutter's own position, so both signals share a
time base and a lag between them is a genuine temporal offset.

METHOD. Both signals are high-pass filtered before the lag is estimated. The
commanded path varies with a period of about 11.7 s, and over a shift of a few
hundred milliseconds the correlation of so slow a signal barely changes, so the
peak is undetermined and the search returns its own boundary -- a first attempt
produced 583 ms, which is 35 of a permitted 40 steps. Removing the slow
component leaves the 0.3-1 Hz band, where the closed loop actually has energy
and where a lag of order 100 ms is resolvable. Estimates whose correlation peak
is not clearly defined are discarded rather than reported.

Sign convention: a POSITIVE lag means the second signal follows the first.

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
PLANT_DEAD_TIME_MS = 117.0        # open-loop, Table 7.2


def _highpass(x, cutoff_hz=0.20, fs=CONTROL_HZ):
    """Remove everything slower than the closed loop's own dynamics."""
    k = max(int(fs / cutoff_hz), 3)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ext = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    slow = np.convolve(ext, np.ones(k) / k, mode="valid")[:x.size]
    return x - slow


def lag_of(drive, resp, max_lag=24, min_prominence=0.05):
    """Signed lag maximising correlation; negative means resp LEADS drive.

    Returns (lag, peak, prominence). A flat correlation curve has a peak that
    means nothing, so the prominence of the maximum over the mean is returned
    and the caller discards weak ones.
    """
    d = np.diff(_highpass(drive))
    r = np.diff(_highpass(resp))
    d = d - d.mean()
    r = r - r.mean()
    cs = []
    lags = list(range(-max_lag, max_lag + 1))
    for L in lags:
        if L >= 0:
            a, b = (d[:len(d) - L] if L else d), r[L:]
        else:
            a, b = d[-L:], r[:len(r) + L]
        n = min(len(a), len(b))
        if n < 64:
            cs.append(-2.0)
            continue
        cs.append(float(np.dot(a[:n], b[:n]) /
                        (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-12)))
    cs = np.array(cs)
    i = int(np.argmax(cs))
    prom = float(cs[i] - np.median(cs))
    if abs(lags[i]) == max_lag:      # railed on the boundary: not a peak
        return None, cs[i], prom
    return lags[i], cs[i], prom
    best, best_c = 0, -2.0
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            a, b = d[:len(d) - L] if L else d, r[L:]
        else:
            a, b = d[-L:], r[:len(r) + L]
        n = min(len(a), len(b))
        if n < 64:
            continue
        c = float(np.dot(a[:n], b[:n]) /
                  (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-12))
        if c > best_c:
            best, best_c = L, c
    return best, best_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mesh", default="1337")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    report = {"plant_dead_time_ms": PLANT_DEAD_TIME_MS, "stacks": {}}
    print(f"\n  plant dead time, open loop: {PLANT_DEAD_TIME_MS:.0f} ms\n")
    print(f"  {'stack':<20}{'achieved lag':>14}{'command lead':>15}")
    for key, label in STACKS:
        p = a.root / f"{key}_m{a.mesh}.npz"
        if not p.is_file():
            continue
        d = np.load(p)
        ref = d["reference_y"][:, :, 0]
        cut = d["cutter_y"][:, :, 0]
        cmd = d["valve_cmd"][:, :, 0]
        ach, lead = [], []
        for e in range(ref.shape[1]):
            if ref[:, e].std() < 1e-6:      # a flat reference carries no phase
                continue
            L, c, pr = lag_of(ref[:, e], cut[:, e])
            if L is not None and pr >= 0.05:
                ach.append(L)
            for i in range(d["valve_cmd"].shape[2]):
                L2, c2, pr2 = lag_of(ref[:, e], d["valve_cmd"][:, e, i])
                if L2 is not None and pr2 >= 0.05:
                    lead.append(L2)
        am = float(np.median(ach)) if ach else float("nan")
        lm = float(np.median(lead)) if lead else float("nan")
        print(f"  {label:<20}{1000*am/CONTROL_HZ:>11.1f} ms{1000*lm/CONTROL_HZ:>12.1f} ms")
        report["stacks"][label] = {
            "achieved_lag_ms": 1000 * am / CONTROL_HZ,
            "command_lead_ms": 1000 * lm / CONTROL_HZ,
            "n_achieved": len(ach), "n_lead": len(lead)}

    print("\n  A stack that had NOT compensated would show an achieved lag near")
    print(f"  the {PLANT_DEAD_TIME_MS:.0f} ms plant dead time; one that had would show it near zero.")
    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
