#!/usr/bin/env python3
"""Turn per-step traces into the three online-adaptation results.

Consumes npz traces written by `TerrainSolePolicyEnv.save_adaptation_trace`
(enable with `cfg.adaptation_trace_path`). Distance-binned reports cannot
resolve any of this: their finest bin is 3.25 m while the AAM warmup spans
~0.29 m of travel.

Three questions, each with the control that makes it falsifiable:

1. WARMUP SEPARATION
   The AAM is invalid for its first `history_lag` model ticks. Arms should
   therefore coincide before the AAM becomes valid and separate afterwards.
   Separation that exists *before* validity is not adaptation -- it is a
   confound (different seeds, different initial conditions), and this script
   reports it as such rather than crediting it.

2. STEP RESPONSE
   After a mid-episode plant change: does z_hat move, does error spike, does
   error recover? Recovery is measured against the FROZEN arm, which holds a
   pre-change estimate. Frozen supplies the same input dimensionality and
   magnitude, so a difference isolates online updating from mere conditioning.

3. RESPONSE LAG
   z_hat should react roughly one history window after the change. Comparing
   L5/L20/L40 turns a performance delta into a mechanism result: if the lag
   scales with the window, the recovery is being driven by the estimator.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONTROL_HZ = 60.0
# Arms whose latent actually updates during the episode. Only these can carry
# an online-adaptation claim; the rest are controls.
UPDATING_ARMS = {"estimated", "true", "A4_full", "full"}
# Below this relative difference the two arms are called indistinguishable
# rather than one being declared better.
NEGLIGIBLE_REL = 0.02


def load(path: Path) -> dict:
    with np.load(path) as d:
        out = {k: d[k] for k in d.files}
    return out


def series(tr: dict, key: str) -> np.ndarray | None:
    """(steps, envs) mean over the field width, or None if absent."""
    if key not in tr:
        return None
    a = tr[key]
    return a.mean(axis=2) if a.ndim == 3 else a


def step_index(tr: dict) -> int | None:
    """First step at which the scheduled plant change fired.

    Returning None means the perturbation was configured but never reached.
    That has a specific cause worth naming: the rollout ends when the truck
    reaches the end of its 26 m path, which at 5 km/h is 18.7 s -- well short
    of the 47.3 s episode length -- so a step scheduled after that never fires
    and every arm silently runs on an unperturbed plant.
    """
    st = series(tr, "theta_stepped")
    if st is None or not st.any():
        return None
    return int(np.argmax(st.max(axis=1) > 0.5))


def summarise(name: str, tr: dict, warmup_steps: int, metric: str) -> dict:
    err = series(tr, metric)
    if err is None:
        raise SystemExit(f"{name}: trace has no field {metric!r}")
    # Errors are recorded SIGNED. Averaging them signed lets a large
    # oscillation cancel to near zero and report as excellent tracking, so
    # every summary below is on the magnitude.
    err = np.abs(err)
    # Prefer the RAW estimate. `z_hat` is what the controller received, so in
    # the routing arms that zero it, reading `z_hat` would report "the latent
    # never moved" -- describing the ablation rather than the estimator.
    z = series(tr, "z_hat_raw")
    if z is None:
        z = series(tr, "z_hat")
    k = step_index(tr)
    n = err.shape[0]
    out = {
        "steps": int(n),
        "envs": int(err.shape[1]),
        "mean_error": float(np.nanmean(err)),
        "pre_warmup_error": float(np.nanmean(err[:warmup_steps])) if n > warmup_steps else float("nan"),
        "post_warmup_error": float(np.nanmean(err[warmup_steps:])) if n > warmup_steps else float("nan"),
        "step_index": k,
    }
    if k is not None:
        w = max(int(0.5 * CONTROL_HZ), 1)          # half-second windows
        pre = err[max(k - w, 0):k]
        spike = err[k:k + w]
        tail = err[max(n - w, 0):]
        out.update({
            "pre_step_error": float(np.nanmean(pre)) if pre.size else float("nan"),
            "spike_error": float(np.nanmean(spike)) if spike.size else float("nan"),
            "final_error": float(np.nanmean(tail)) if tail.size else float("nan"),
        })
        if z is not None:
            zpre = np.nanmean(z[max(k - w, 0):k], axis=0)
            # Measure the SETTLED shift, not an immediate one. The latent is
            # expected to respond about one history window after the change,
            # so a short post-window closes before it has moved and reports a
            # spurious zero shift for an arm that did adapt.
            zpost = np.nanmean(z[max(n - w, 0):], axis=0)
            out["z_shift_at_step"] = float(np.nanmean(np.abs(zpost - zpre)))
            # lag: first step after the change at which |z - z_pre| exceeds
            # a quarter of its eventual shift
            dz = np.abs(np.nanmean(z[k:], axis=1) - np.nanmean(zpre))
            final = np.nanmax(dz) if dz.size else 0.0
            if final > 0:
                crossed = np.argmax(dz > 0.25 * final)
                out["z_response_lag_s"] = float(crossed / CONTROL_HZ)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", action="append", required=True, metavar="NAME=PATH",
                   help="named trace, repeatable, e.g. --trace estimated=est.npz")
    p.add_argument("--metric", default="lateral_error")
    p.add_argument("--history-lag", type=int, default=20,
                   help="AAM history lag in MODEL ticks (50 ms each)")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "results/adaptation_trace.json")
    a = p.parse_args()

    warmup_s = a.history_lag * 0.05
    warmup_steps = int(round(warmup_s * CONTROL_HZ))
    print(f"AAM warmup: {a.history_lag} model ticks = {warmup_s:.2f} s "
          f"= {warmup_steps} control steps\n")

    traces, summaries = {}, {}
    for spec in a.trace:
        if "=" not in spec:
            raise SystemExit(f"--trace expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        traces[name] = load(Path(path))
        summaries[name] = summarise(name, traces[name], warmup_steps, a.metric)

    print(f"=== {a.metric} by arm ===")
    hdr = f"{'arm':12} {'pre-warmup':>12} {'post-warmup':>12} {'mean':>10} {'z shift':>10} {'lag s':>8}"
    print(hdr); print("-" * len(hdr))
    for n, s in summaries.items():
        print(f"{n:12} {s['pre_warmup_error']:>12.5f} {s['post_warmup_error']:>12.5f} "
              f"{s['mean_error']:>10.5f} {s.get('z_shift_at_step', float('nan')):>10.4f} "
              f"{s.get('z_response_lag_s', float('nan')):>8.2f}")

    # 1. warmup separation, with the confound check
    print("\n=== 1. warmup separation ===")
    if len(summaries) >= 2:
        names = list(summaries)
        base = names[0]
        for other in names[1:]:
            dpre = summaries[other]["pre_warmup_error"] - summaries[base]["pre_warmup_error"]
            dpost = summaries[other]["post_warmup_error"] - summaries[base]["post_warmup_error"]
            print(f"  {other} vs {base}:  pre {dpre:+.5f}   post {dpost:+.5f}")
            if abs(dpre) > 0.25 * abs(dpost) and abs(dpost) > 0:
                print("    WARNING: arms already differ before the AAM is valid.")
                print("    That separation cannot be adaptation -- suspect a confound")
                print("    (unpaired seeds, differing initial conditions).")
            elif abs(dpost) > abs(dpre):
                print("    separation appears only after the AAM becomes valid.")

    # 2. step response vs the frozen control
    print("\n=== 2. step response ===")
    stepped = {n: s for n, s in summaries.items() if s.get("step_index") is not None}
    if not stepped:
        print("  no scheduled plant change fired in these traces.")
        elapsed = {n: float(np.nanmax(series(t, "theta_elapsed_s")))
                   for n, t in traces.items() if "theta_elapsed_s" in t}
        for n, e in elapsed.items():
            print(f"    {n}: episode reached only {e:.1f}s")
        print("  If a step time was set, it was scheduled past the end of the")
        print("  SCORED window: the rollout stops when the truck completes its")
        print("  26 m path (18.7 s at 5 km/h), not at the 47.3 s episode length.")
        print("  Every arm then ran on an unperturbed plant -- the contrasts")
        print("  below are NOT an online-adaptation result.")
    else:
        for n, s in stepped.items():
            rec = s["spike_error"] - s["final_error"]
            print(f"  {n:12} pre {s['pre_step_error']:.5f} -> spike {s['spike_error']:.5f} "
                  f"-> final {s['final_error']:.5f}   (recovered {rec:+.5f})")
        if "frozen" in stepped and len(stepped) > 1:
            fz = stepped["frozen"]
            for n, s in stepped.items():
                if n == "frozen":
                    continue
                adv = fz["final_error"] - s["final_error"]
                rel = adv / max(abs(fz["final_error"]), 1e-9)
                print(f"\n  {n} vs frozen, final error: {adv:+.5f} ({rel:+.1%})")
                # Only an arm whose latent actually UPDATES can have its
                # advantage attributed to online estimation. Comparing a
                # zero/shuffled/mean arm against frozen says nothing about
                # updating -- those arms have no live estimate to update --
                # and reporting it as such was this script's own bug.
                if n not in UPDATING_ARMS:
                    print(f"    -> {n!r} is not an updating arm; this difference")
                    print("       says nothing about online adaptation. It compares a")
                    print("       degraded latent against a held one, nothing more.")
                elif abs(rel) < NEGLIGIBLE_REL:
                    print(f"    -> INDISTINGUISHABLE ({abs(rel):.1%} < "
                          f"{NEGLIGIBLE_REL:.0%}). Holding a PRE-change estimate")
                    print("       costs nothing, so nothing here supports an online")
                    print("       adaptation claim. Either the latent does not encode")
                    print("       this perturbation, or the controller is insensitive")
                    print("       to it. Check the sham-step control before concluding.")
                elif adv > 0:
                    print("    -> the updating arm ends closer to target. Consistent")
                    print("       with online estimation -- but confirm the latent")
                    print("       moves MORE across a real step than a sham one.")
                else:
                    print("    -> no advantage over a held estimate. The controller was")
                    print("       robust anyway, or the latent is not tracking the change.")
        elif "frozen" not in stepped:
            print("\n  NOTE: no `frozen` arm supplied. Without it a recovery cannot be")
            print("  separated from simply having extra input dimensions.")

    # 3. lag vs window
    print("\n=== 2b. is the z shift caused by the STEP? ===")
    sham = [n for n in summaries if "sham" in n.lower() or "nostep" in n.lower()]
    if not sham:
        print("  UNKNOWN -- no sham-step arm supplied. z_hat drifts as the")
        print("  trajectory evolves, so a shift measured across the step time is")
        print("  not by itself evidence that the latent tracked the plant change.")
        print("  Add an arm with an identical schedule and step_alpha_scale=1.0.")
    else:
        ref = summaries[sham[0]].get("z_shift_at_step")
        for n, s in summaries.items():
            if n in sham or "z_shift_at_step" not in s:
                continue
            if ref:
                print(f"  {n:12} shift {s['z_shift_at_step']:.4f} vs sham "
                      f"{ref:.4f}  ratio {s['z_shift_at_step']/ref:.2f}x")

    print("\n=== 3. response lag ===")
    lags = {n: s["z_response_lag_s"] for n, s in summaries.items()
            if "z_response_lag_s" in s}
    if lags:
        for n, v in lags.items():
            print(f"  {n:12} {v:.2f} s   (one history window = {warmup_s:.2f} s)")
        print("  Rerun at L5/L20/L40; a lag that scales with the window is")
        print("  mechanism evidence that the estimator drives the recovery.")
    else:
        print("  not measurable without a scheduled plant change")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "format": "adaptation_trace_v1",
        "metric": a.metric,
        "history_lag_model_ticks": a.history_lag,
        "warmup_control_steps": warmup_steps,
        "arms": summaries,
    }, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
