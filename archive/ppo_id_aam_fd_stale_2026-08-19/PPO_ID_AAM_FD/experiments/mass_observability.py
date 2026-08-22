#!/usr/bin/env python3
"""Is EE mass physically observable in the signals the AAM receives?

Motivation
----------
The AAM row is ``[joint_q(5), joint_qdot(5), dP(4), previous_valve(4)]``.
That is enough, in principle, to identify inertia: differencing ``qdot`` gives
acceleration, ``dP`` gives cylinder force, and ``m = F/a``.  So "mass is not in
the history" cannot be asserted from a failed regression alone -- a network can
fail for many reasons.

This probe removes the network.  It measures, per run, the *plant gain* from
cylinder pressure to joint acceleration::

    qddot (5)  =  A . dP (4)  +  b

``A`` is an effective inverse-inertia matrix.  If ``mass_scale`` genuinely
changes the plant, ``A`` must shrink as mass grows -- that is Newton's second
law, not a modelling assumption.  Correlating ``A`` against the true factors
across runs therefore tests observability directly.

``seal`` is carried as a POSITIVE CONTROL.  Seal is known to be strongly
encoded (held-out R2 0.956 through the SD latent).  If this method recovers a
seal signature but no mass signature, the method works and the mass null is
real.  If it recovers neither, the method is broken and proves nothing.

CPU only.  No model, no training, no simulator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "AAM_module/datasets/rom_deployment_stage31_v1_frozen"
FACTORS = ["flow", "compliance", "seal", "pressure", "mass_scale", "com_z"]
ROM_DT = 0.05
Q = slice(0, 5)
QD = slice(5, 10)
DP = slice(10, 14)
UV = slice(14, 18)


def run_gain(feat: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return (flattened dP->qddot gain, magnitude diagnostics) for one run."""
    qd = feat[:, QD]
    dp = feat[:, DP]
    uv = feat[:, UV]
    qdd = np.diff(qd, axis=0) / ROM_DT           # (n-1, 5)
    dp_m = dp[:-1]
    # regress qddot on [dP, valve, 1] -- valve included so the gain reflects
    # response to pressure at fixed command, not command-following alone.
    X = np.hstack([dp_m, uv[:-1], np.ones((len(dp_m), 1))])
    A, *_ = np.linalg.lstsq(X, qdd, rcond=None)  # (9, 5)
    gain = A[:4, :].ravel()                      # dP -> qddot block, 20 values
    diag = {
        "abs_qddot_mean": float(np.abs(qdd).mean()),
        "abs_qddot_p95": float(np.percentile(np.abs(qdd), 95)),
        "abs_qdot_mean": float(np.abs(qd).mean()),
        "abs_dp_mean": float(np.abs(dp).mean()),
        "gain_norm": float(np.linalg.norm(gain)),
    }
    return gain, diag


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "results/mass_observability.json")
    a = p.parse_args()

    aam_root = a.root / "aam/dataset_multilag"
    runs_info = json.loads((a.root / "splits/runs_info.json").read_text())

    gains, keys, diags, ids = [], [], [], []
    for d in sorted(aam_root.iterdir()):
        f = d / "features.npz"
        info = runs_info.get(d.name)
        if not f.is_file() or info is None:
            continue
        with np.load(f) as z:
            feat = z["features"].astype(np.float64)
        if feat.shape[0] < 50:
            continue
        g, diag = run_gain(feat)
        if not np.isfinite(g).all():
            continue
        gains.append(g)
        keys.append(info["key"])
        diags.append(diag)
        ids.append(d.name)

    G = np.array(gains)          # (N, 20)
    K = np.array(keys)           # (N, 6)
    print(f"runs analysed: {len(G)}   gain features per run: {G.shape[1]}")

    print("\n--- signal magnitudes (is there any acceleration to work with?) ---")
    for k in ("abs_qdot_mean", "abs_qddot_mean", "abs_qddot_p95", "abs_dp_mean"):
        v = np.array([d[k] for d in diags])
        print(f"  {k:16} mean={v.mean():.6g}  p05={np.percentile(v,5):.6g}  p95={np.percentile(v,95):.6g}")

    print("\n--- can each factor be read off the measured plant gain? ---")
    print("    (held-out R2, leave-one-run-out ridge on the dP->qddot gain)")
    results = {}
    N = len(G)
    Gs = (G - G.mean(0)) / G.std(0).clip(min=1e-12)
    for j, name in enumerate(FACTORS):
        y = K[:, j]
        if y.std() < 1e-9:
            continue
        preds = np.zeros(N)
        for i in range(N):
            m = np.ones(N, bool)
            m[i] = False
            X = np.hstack([Gs[m], np.ones((m.sum(), 1))])
            lam = 1.0
            A = X.T @ X + lam * np.eye(X.shape[1])
            w = np.linalg.solve(A, X.T @ y[m])
            preds[i] = np.hstack([Gs[i], 1.0]) @ w
        ss_res = ((y - preds) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        corr = float(np.corrcoef(y, preds)[0, 1])
        results[name] = {"r2_loo": float(r2), "corr": corr}
        tag = "  <-- POSITIVE CONTROL" if name == "seal" else ""
        print(f"  {name:12} R2={r2:+.3f}  corr={corr:+.3f}{tag}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "format": "mass_observability_v1",
        "method": "per-run linear plant gain dP->qddot, leave-one-run-out ridge decode of true factors",
        "n_runs": int(N),
        "magnitudes": {k: float(np.mean([d[k] for d in diags]))
                       for k in diags[0]},
        "factor_decode": results,
    }, indent=2))
    print(f"\nwrote {a.out}")

    print("\n--- reading ---")
    seal = results.get("seal", {}).get("r2_loo", float("nan"))
    mass = results.get("mass_scale", {}).get("r2_loo", float("nan"))
    if seal > 0.3 and mass < 0.1:
        print("  method works (seal recovered) but mass is NOT in the plant gain")
        print("  -> mass is genuinely weakly observable from [q, qdot, dP, valve]")
    elif seal <= 0.3:
        print("  positive control FAILED -- this method cannot detect a known-encoded")
        print("  factor, so it says nothing about mass. Do not conclude from it.")
    else:
        print("  mass IS present in the measured plant gain")
        print("  -> earlier 'mass not in history' conclusion was wrong; the")
        print("     network probe underfit rather than the signal being absent")


if __name__ == "__main__":
    main()
