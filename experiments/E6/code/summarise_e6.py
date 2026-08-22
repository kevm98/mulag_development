#!/usr/bin/env python3
"""E6: is it the latent's BANDWIDTH that hurts the closed loop, or its content?

E4 ranked four latents and the deployed estimate came last. Chapter 7 attributes
that to the estimate's time-variation, but E4's arms differ in content AND in
variability, so the attribution is an inference rather than a measurement.

E6 holds content fixed and sweeps bandwidth alone: every arm is the same
checkpoint reading the same estimate, low-passed on the policy path only. The
two zero-bandwidth arms are the discriminator -- `frozen` holds the CORRECT
content, `mean` holds none -- so a gap between them is content and a gap along
the lpf series is bandwidth.

Unit is the (mesh, family) cell; the evaluation is deterministic and the seed is
not replication.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Ordered by policy-path bandwidth, high to low. `mean` sits with `frozen` at
# zero and differs from it only in content.
# (key, label, f_c, tau). The sweep axis is tau = 1/(2*pi*f_c); raw is
# unfiltered and latched is tau -> infinity in effect, so both are endpoints
# rather than points on the tau axis.
ARMS = [
    ("bw_estimated", "raw (unfiltered)",       None,   None),
    ("bw_lpf_5p0",   "lpf tau 0.032 s",        5.0,    0.032),
    ("bw_lpf_1p0",   "lpf tau 0.159 s",        1.0,    0.159),
    ("bw_lpf_0p64",  "lpf tau 0.250 s",        0.6366, 0.250),
    ("bw_lpf_0p32",  "lpf tau 0.500 s",        0.3183, 0.500),
    ("bw_lpf_0p2",   "lpf tau 0.796 s",        0.2,    0.796),
    ("bw_lpf_0p04",  "lpf tau 3.979 s",        0.04,   3.979),
    ("bw_frozen",    "latched (no update)",    None,   None),
    ("bw_mean",      "mean (no plant info)",   None,   None),
]
# Ordered by decreasing policy-path bandwidth, for the response-shape check.
LPF_SERIES = ["bw_estimated", "bw_lpf_5p0", "bw_lpf_1p0", "bw_lpf_0p64",
              "bw_lpf_0p32", "bw_lpf_0p2", "bw_lpf_0p04", "bw_frozen"]
METRICS = {"mean_abs_lateral_error_m": ("lateral error (m)", True),
           "mean_abs_cut_height_error_m": ("cut-height error (m)", True),
           "ground_contact_fraction": ("ground contact", False)}


def cells(root: Path, arm: str, meshes: list[str]):
    out = {}
    for m in meshes:
        p = root / f"{arm}_m{m}.json"
        if not p.is_file():
            return None
        blk = json.loads(p.read_text())["per_mesh"].get(m)
        if not blk:
            return None
        for fam in blk.get("path_family_results", []):
            out[(m, fam["family"])] = fam
    return out


def check_manifest(root: Path, arm: str, meshes: list[str], want_hz: float):
    """The arm must have RUN at the cutoff its name claims.

    A silent fallback to `estimated` is the failure mode that would make this
    whole experiment report a flat line, so it is checked rather than trusted.
    """
    for m in meshes:
        rec = json.loads((root / f"{arm}_m{m}.json").read_text())
        man = rec.get("manifest", rec)
        got = float(man.get("online_aam_policy_latent_lpf_hz", 0.0))
        mode = str(man.get("online_aam_latent_mode", ""))
        ctl = str(man.get("online_aam_controller_latent_mode", ""))
        if ctl and ctl != "estimated":
            raise SystemExit(
                f"{arm} m{m}: controller latent is {ctl!r}, not 'estimated'. "
                "E6 requires the adapter to keep the full-rate estimate; this "
                "run reintroduces the confound the experiment removes."
            )
        if mode == "lpf" and want_hz not in (0.0, float("inf")):
            if abs(got - want_hz) > 1e-9:
                raise SystemExit(
                    f"{arm} m{m}: ran at {got} Hz but the arm is named "
                    f"{want_hz} Hz"
                )


def boot(d, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def contrast(name, a, b, lower_better, n):
    """Paired contrast reported on the mean AND the median.

    The mean alone is not enough here: these distributions are strongly
    right-skewed (a mean five times the median is normal in this task), so a
    mean can move twenty per cent while typical behaviour does not move at all.
    The median PAIRED difference and a Wilcoxon signed-rank test are what say
    whether the middle of the distribution shifted. A ratio of medians is NOT a
    substitute: it discards the pairing and can read flat when every cell moved.
    """
    d = a - b
    better = int((d < 0).sum()) if lower_better else int((d > 0).sum())
    lo, hi = boot(d)
    excl = (hi < 0) if lower_better else (lo > 0)
    rel = 100 * (a.mean() / b.mean() - 1) if b.mean() else float("nan")
    med = float(np.median(d))
    med_pct = 100 * float(np.median(d / np.where(b == 0, np.nan, b)))
    try:
        from scipy import stats as _st
        wp = float(_st.wilcoxon(a, b).pvalue)
    except Exception:
        wp = float("nan")
    print(f"      {name:<24}mean{rel:+7.1f}%  med{med_pct:+7.1f}%  {better}/{n}  "
          f"CI [{lo:+.5f}, {hi:+.5f}] {'RES' if excl else '   '}  W p={wp:.3f}")
    return {"rel_pct": float(rel), "median_diff": med, "median_pct": med_pct,
            "better": better, "n": n, "ci": [lo, hi], "resolved": bool(excl),
            "wilcoxon_p": wp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--meshes", default="1337 17 909")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    meshes = a.meshes.split()

    data = {k: cells(a.root, k, meshes) for k, _, _, _ in ARMS}
    missing = [k for k, v in data.items() if v is None]
    if missing:
        raise SystemExit(f"incomplete grid, refusing to summarise: {missing}")
    for k, _, hz, _tau in ARMS:
        check_manifest(a.root, k, meshes, hz)
    keys = sorted(set.intersection(*(set(v) for v in data.values())))
    n = len(keys)
    print(f"\n  {n} paired (mesh, family) cells per arm, one checkpoint\n")

    report = {"n_cells": n, "meshes": meshes, "metrics": {}}
    for key, (name, lower_better) in METRICS.items():
        vals = {k: np.array([data[k][c][key] for c in keys]) for k, _, _, _ in ARMS}
        print(f"  {name}   (bandwidth high -> low)")
        for k, label, _, _ in ARMS:
            print(f"    {label:<28}{vals[k].mean():10.5f}")
        rec = {"per_arm_mean": {k: float(vals[k].mean()) for k, _, _, _ in ARMS}}

        # Validate the instrument before reading the result. If the filter were
        # inert, every lpf arm would score bit-identically to bw_estimated and
        # the sweep would report a flat line that means nothing at all. That is
        # indistinguishable from a true null unless it is checked here.
        inert = [k for k in LPF_SERIES[1:-1]
                 if np.array_equal(vals[k], vals["bw_estimated"])]
        if inert:
            raise SystemExit(
                f"{name}: arms {inert} are bit-identical to bw_estimated. The "
                "low-pass did nothing -- check that --online_aam_policy_latent_"
                "lpf_hz reached the config, not that bandwidth does not matter."
            )

        print("    vs the deployed unlimited-bandwidth arm:")
        rec["vs_estimated"] = {
            k: contrast(label, vals[k], vals["bw_estimated"], lower_better, n)
            for k, label, _, _ in ARMS if k != "bw_estimated"
        }

        # The discriminator: same bandwidth, different content.
        print("    content at zero bandwidth:")
        rec["frozen_vs_mean"] = contrast(
            "frozen vs mean (content only)",
            vals["bw_frozen"], vals["bw_mean"], lower_better, n)

        # The hypothesis is an OPTIMUM at intermediate bandwidth, not a
        # monotone trend, so monotonicity is the wrong test: an optimum is
        # non-monotone by construction and a rank correlation would count that
        # against the very hypothesis it is meant to check. Report the shape.
        series = [(k, float(vals[k].mean())) for k in LPF_SERIES]
        best_k, best_v = min(series, key=lambda kv: kv[1]) if lower_better \
            else max(series, key=lambda kv: kv[1])
        idx = [k for k, _ in series].index(best_k)
        interior = 0 < idx < len(series) - 1
        print(f"      response shape: best = {best_k} ({best_v:.5f}); "
              f"{'INTERIOR optimum' if interior else 'optimum at an endpoint'}")
        print("        " + "  ".join(f"{v:.5f}" for _, v in series))
        rec["response_by_decreasing_bandwidth"] = {k: v for k, v in series}
        rec["best_arm"] = best_k
        rec["interior_optimum"] = bool(interior)
        report["metrics"][key] = rec
        print()

    # Mechanism variables. These are per-run, not per-cell, so they are
    # reported as an axis for the response above rather than tested.
    print("  mechanism variables (mean over meshes)")
    print(f"    {'arm':<26}{'Var(z_hat) seen by actor':>26}{'reversals/s':>14}")
    report["mechanism"] = {}
    for k, label, _, tau in ARMS:
        vv, rr = [], []
        for m in meshes:
            rec_m = json.loads((a.root / f"{k}_m{m}.json").read_text())
            vv.append(float(rec_m.get("policy_latent_temporal_var", 0.0)))
            rr.append(float(rec_m.get("command_reversals_per_s", 0.0)))
        v, r = float(np.mean(vv)), float(np.mean(rr))
        print(f"    {label:<26}{v:>26.5f}{r:>14.2f}")
        report["mechanism"][k] = {"policy_latent_temporal_var": v,
                                  "command_reversals_per_s": r, "tau_s": tau}
    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
