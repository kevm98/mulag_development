#!/usr/bin/env python3
"""Where in the excitation envelope is the plant legible from z_hat?

This started as a test of online refinement -- does z_hat get better as the
episode proceeds? On this collection that question CANNOT be asked this way,
and the script says so rather than reporting the trend it would otherwise
produce. The open-loop plan sweeps excitation amplitude monotonically across
the episode (0.05 -> 0.15 -> 0.35 -> 0.7, one tier per quarter), so episode
time and excitation amplitude are the same variable. Any "improves over time"
or "decays over time" reading would be an amplitude effect wearing a clock.

Two further reasons the refinement framing does not apply:

  * the AAM consumes a FIXED 20-tick (1.0 s) causal window and carries no state
    between ticks, so it cannot accumulate evidence over an episode by
    construction; it re-estimates from the last second, every second.
  * the large-amplitude tiers are precisely the ones the sample mask flags for
    actuator saturation, and a saturated response is plant-independent.

So the question this actually answers is the useful one: at which excitation
amplitude does z_hat carry plant identity? The claim-1 test (does the latent's
within-episode variation do any work?) lives in the latent ablation written by
`train_sd_aam_e2e.py --eval-only`, which compares the online latent against a
per-run constant and against a shuffled one on the same trained model.

Decoders are fit on TRAIN runs and scored on HELD-OUT runs. `seal` is the
POSITIVE CONTROL -- if it collapses, the decode is broken and nothing else in
the output means anything.

CPU only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "AAM_module"))
from aam_data import load_split  # noqa: E402

FACTORS = ["flow", "compliance", "seal", "pressure", "mass_scale", "com_z"]
CONTROL, CONTROL_MIN = "seal", 0.5


def load_traces(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """run_id -> (ticks ascending, z_hat rows)."""
    by_run = json.loads(path.read_text())["by_run"]
    out = {}
    for rid, ticks in by_run.items():
        keys = sorted(ticks, key=int)
        out[int(rid)] = (np.array([int(k) for k in keys]),
                         np.array([ticks[k] for k in keys], dtype=float))
    return out


def load_factors(root: Path) -> dict[int, np.ndarray]:
    """run_id -> the true 6-factor key, in FACTORS order."""
    info = json.loads((root / "splits/runs_info.json").read_text())
    return {int(k): np.asarray(v["key"], dtype=float) for k, v in info.items()}


def amplitude_tiers(raw_root: Path, run_id: int) -> list[tuple[float, int, int]]:
    """(amplitude, start_control_step, end_control_step) per contiguous tier.

    Read from the collection's own segment schedule rather than assumed, so a
    re-planned collection does not silently invalidate the grouping.
    """
    meta = json.loads((raw_root / f"runs/{run_id:05d}/metadata.json").read_text())
    spans: dict[float, list[int]] = {}
    for s in meta["segment_ids"]:
        if s.get("kind") != "probe":
            continue
        m = re.search(r"_a([0-9.]+)_", s["name"])
        if not m:
            continue
        amp = float(m.group(1))
        lo, hi = spans.setdefault(amp, [s["start"], s["end"]])
        spans[amp] = [min(lo, s["start"]), max(hi, s["end"])]
    return [(a, *spans[a]) for a in sorted(spans)]


def ridge_r2(Xtr, ytr, Xte, yte, lam=1e-3) -> float:
    """Held-out R^2 of an affine decode, measured against the TEST mean."""
    Xtr, Xte = np.c_[Xtr, np.ones(len(Xtr))], np.c_[Xte, np.ones(len(Xte))]
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    ss_res = float(((yte - Xte @ w) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True,
                    help="frozen dataset (splits + z_hat tick base)")
    ap.add_argument("--raw-root", type=Path, required=True,
                    help="raw collection, for the per-run segment schedule")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    traces = load_traces(a.model_dir / "zhat_all.json")
    factors = load_factors(a.root)
    tr = [r for r in load_split(a.root, "train") if r in traces and r in factors]
    te = [r for r in (list(load_split(a.root, "val")) + list(load_split(a.root, "test")))
          if r in traces and r in factors]
    if len(te) < 8:
        raise SystemExit(f"only {len(te)} held-out runs; decode would be noise")

    tiers = amplitude_tiers(a.raw_root, tr[0])
    if not tiers:
        raise SystemExit("no amplitude tiers found in the segment schedule")

    # z_hat is emitted every 3rd control step (60 Hz control, 20 Hz model tick)
    # but its keys are CONTROL-step indices, so the segment schedule -- also in
    # control steps -- indexes it directly. Verified against run 0: ticks run
    # 57..11037 against num_control_steps 11040.
    def tier_mean(run: int, lo: int, hi: int) -> np.ndarray | None:
        ticks, z = traces[run]
        sel = (ticks >= lo) & (ticks < hi)
        return z[sel].mean(0) if sel.any() else None

    Ytr = np.array([factors[r] for r in tr])
    Yte = np.array([factors[r] for r in te])

    print(f"\n  runs: {len(tr)} train / {len(te)} held out")
    print("  NOTE: tier and episode time are the same axis in this collection; "
          "these are\n        amplitude effects, NOT a within-episode trend.")
    print("\n  held-out R^2 of z_hat -> true factor, by excitation amplitude")
    print("   " + "amp".ljust(11) + "".join(f"{f:>12}" for f in FACTORS)
          + f"{'mean':>12}")

    rows, table = [], {}
    for amp, lo, hi in tiers:
        Xtr = [tier_mean(r, lo, hi) for r in tr]
        Xte = [tier_mean(r, lo, hi) for r in te]
        if any(x is None for x in Xtr + Xte):
            print(f"   {amp:<11.2f} incomplete coverage, skipped")
            continue
        r2 = np.array([ridge_r2(np.array(Xtr), Ytr[:, j],
                                np.array(Xte), Yte[:, j])
                       for j in range(len(FACTORS))])
        table[amp] = r2
        rows.append((amp, r2))
        print(f"   {amp:<11.2f}" + "".join(f"{v:12.3f}" for v in r2)
              + f"{np.nanmean(r2):12.3f}")

    if not rows:
        raise SystemExit("no tier had complete coverage")

    ctrl = FACTORS.index(CONTROL)
    best_amp, best_r2 = max(rows, key=lambda kv: kv[1][ctrl])[0], \
        float(max(r[1][ctrl] for r in rows))
    worst_r2 = float(min(r[1][ctrl] for r in rows))
    ok = best_r2 >= CONTROL_MIN

    # How far z_hat swings inside one episode, in units of how far plants differ.
    per_run_mean = np.array([traces[r][1].mean(0) for r in tr + te])
    spread = per_run_mean.std(0).clip(min=1e-12)
    swing = np.array([traces[r][1].std(0) for r in tr + te]).mean(0) / spread

    print(f"\n  positive control ({CONTROL}): best {best_r2:.3f} at amplitude "
          f"{best_amp:.2f}, worst {worst_r2:.3f}   "
          f"{'OK' if ok else 'FAILED -- decode is broken, results are void'}")
    print("  within-episode swing / between-plant spread: "
          + ", ".join(f"dim{i} {v:.2f}" for i, v in enumerate(swing)))
    if ok and worst_r2 < CONTROL_MIN:
        print(f"\n  z_hat identifies the plant only at LOW excitation. Identity "
              f"is lost by\n  amplitude {max(t[0] for t in tiers):.2f}, which is "
              f"where the actuator saturates and the\n  response stops depending "
              f"on which plant produced it.")
    if float(swing.max()) > 1.0:
        print("  z_hat varies MORE within an episode than plants differ from "
              "each other,\n  so a per-run mean is a lossy summary of it -- see "
              "latent_ablation.json.")

    if a.out:
        a.out.write_text(json.dumps({
            "model_dir": str(a.model_dir),
            "caveat": "excitation amplitude is confounded with episode time in "
                      "this collection; not a refinement test",
            "factors": FACTORS,
            "by_amplitude": {str(k): v.tolist() for k, v in table.items()},
            "control": CONTROL, "control_best_r2": best_r2,
            "control_best_amplitude": best_amp,
            "control_worst_r2": worst_r2, "control_passed": bool(ok),
            "within_episode_swing_over_spread": swing.tolist(),
            "n_train": len(tr), "n_heldout": len(te),
        }, indent=2))
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
