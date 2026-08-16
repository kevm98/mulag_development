#!/usr/bin/env python3
"""Phase 3 gate: is this latent fit to carry an adaptation claim?

Two properties, BOTH required. Either alone is misleading:

  SEPARATION  z_hat must distinguish different plants. Without it the latent
              carries no plant identity and conditioning on it is noise.

  TRACKING    z_hat must MOVE when the plant changes mid-episode. A latent can
              separate plants perfectly and still be useless for adaptation --
              that is exactly what the old per-run labels produced, a static
              fingerprint that "does not refine online" because it was trained
              not to.

These pull against each other, which is why both are scored. A perfect per-run
fingerprint has zero within-run variance and therefore cannot track; a latent
that chases fast state tracks everything and separates nothing.

`seal` is the POSITIVE CONTROL for separation: it is the factor this pipeline
demonstrably encodes best. If seal fails, the decode is broken and no other
number here means anything.

Nothing downstream -- no ablation, no deployment, no figure -- should use a
latent that has not passed this gate.

CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import FACTOR_KEYS, load_split  # noqa: E402
from tools.train_stage31_aam import AAM  # noqa: E402

CONTROL = "seal"
CONTROL_MIN = 0.5
SEPARATION_MIN = 0.5      # held-out R2 for the control factor
TRACK_RATIO_MIN = 3.0     # post-step shift vs within-segment noise


def load_aam(model_dir: Path):
    metrics = json.loads((model_dir / "metrics.json").read_text())
    norm = metrics["input_normalization"]
    mean = np.asarray(norm["per_position_mean"], dtype=np.float64)
    std = np.asarray(norm["per_position_std"], dtype=np.float64).clip(min=1e-8)
    model = AAM(
        int(metrics["in_dim"]), int(metrics["latent_dim"]),
        hidden=int(metrics.get("hidden", 256)), depth=int(metrics.get("depth", 3)),
        dropout=float(metrics.get("dropout", 0.05)),
        architecture=str(metrics.get("architecture", "cnn")),
        feature_dim=int(metrics.get("feature_dim", 18)),
        lag=int(metrics["lag_ticks"]),
    )
    model.load_state_dict(torch.load(model_dir / "aam.pt", map_location="cpu",
                                     weights_only=False))
    model.eval()
    return model, mean, std, metrics


def predict(model, mean, std, window: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(((window - mean) / std).astype(np.float32))
    with torch.no_grad():
        return torch.cat([model(x[i:i + 8192]) for i in range(0, len(x), 8192)]).numpy()


def r2(y, p, ref):
    return [float(1 - ((y[:, j] - p[:, j]) ** 2).sum()
                  / ((y[:, j] - ref[j]) ** 2).sum()) for j in range(len(FACTOR_KEYS))]


def mlp_decode(Xtr, Ytr, Xva, Yva, Xte, seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 32), nn.SiLU(),
                      nn.Linear(32, len(FACTOR_KEYS)))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-3)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32)
    best, state, bad = float("inf"), None, 0
    for _ in range(4000):
        opt.zero_grad()
        nn.functional.mse_loss(m(T(Xtr)), T(Ytr)).backward()
        opt.step()
        with torch.no_grad():
            vl = float(nn.functional.mse_loss(m(T(Xva)), T(Yva)))
        if vl < best - 1e-7:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 300:
                break
    if state:
        m.load_state_dict(state)
    with torch.no_grad():
        return m(T(Xte)).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--lag", type=int, default=20)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    model, mean, std, metrics = load_aam(a.model_dir)
    aam_root = a.root / "aam" / "dataset_multilag"
    splits = {s: load_split(a.root, s) for s in ("train", "val", "test")}

    per_run, stepped = {}, {}
    for split, ids in splits.items():
        for rid in ids:
            f = aam_root / f"{int(rid):05d}" / f"history_L{a.lag}.npz"
            if not f.is_file():
                continue
            with np.load(f, allow_pickle=False) as d:
                z = predict(model, mean, std, d["window_input"].astype(np.float64))
                e = d["e_target"].astype(np.float64)
                tick = d["window_end_tick"].astype(np.int64)
            per_run[int(rid)] = {"split": split, "z": z, "e": e, "tick": tick}
            if np.ptp(e, axis=0).max() > 0:
                stepped[int(rid)] = True

    if not per_run:
        raise SystemExit(f"no runs found under {aam_root}")
    report = {"model_dir": str(a.model_dir), "n_runs": len(per_run),
              "n_stepped": len(stepped)}

    # ---------------------------------------------------------- SEPARATION
    # Constant-plant runs only: a stepped run has no single plant identity.
    const = {k: v for k, v in per_run.items() if k not in stepped}
    sep = {}
    tr = [k for k, v in const.items() if v["split"] == "train"]
    va = [k for k, v in const.items() if v["split"] == "val"]
    te = [k for k, v in const.items() if v["split"] == "test"]
    if tr and va and te:
        X = {n: np.stack([const[k]["z"].mean(0) for k in ids])
             for n, ids in (("tr", tr), ("va", va), ("te", te))}
        E = {n: np.stack([const[k]["e"][0] for k in ids])
             for n, ids in (("tr", tr), ("va", va), ("te", te))}
        mu, sd = E["tr"].mean(0), E["tr"].std(0).clip(min=1e-9)
        Y = {n: (E[n] - mu) / sd for n in E}
        pred = mlp_decode(X["tr"], Y["tr"], X["va"], Y["va"], X["te"])
        vals = r2(Y["te"], pred, Y["tr"].mean(0))
        sep = {f: float(v) for f, v in zip(FACTOR_KEYS, vals)}
        sep_n = {"train": len(tr), "val": len(va), "test": len(te)}
    else:
        sep_n = {"train": len(tr), "val": len(va), "test": len(te)}
    report["separation"] = {"r2_per_factor": sep, "n_constant_runs": sep_n}

    # ------------------------------------------------------------ TRACKING
    track = []
    for rid in sorted(stepped):
        v = per_run[rid]
        z, e, tick = v["z"], v["e"], v["tick"]
        j = int(np.argmax(np.abs(e - e[0]).max(axis=1) > 0))
        if j <= 0 or j >= len(z) - 1:
            continue
        pre, post = z[:j], z[j:]
        shift = float(np.abs(post.mean(0) - pre.mean(0)).max())
        noise = float(max(pre.std(0).max(), post.std(0).max(), 1e-12))
        # lag: first window after the change whose z exceeds a quarter of the
        # eventual shift; expected ~one history window if the AAM is tracking
        dz = np.abs(z[j:] - pre.mean(0)).max(axis=1)
        final = float(dz.max()) if dz.size else 0.0
        lag = int(np.argmax(dz > 0.25 * final)) if final > 0 else -1
        track.append({"run": rid, "shift": shift, "noise": noise,
                      "ratio": shift / noise, "lag_windows": lag,
                      "change_window": j})
    if track:
        ratios = np.array([t["ratio"] for t in track])
        lags = np.array([t["lag_windows"] for t in track if t["lag_windows"] >= 0])
        report["tracking"] = {
            "n_stepped_scored": len(track),
            "median_shift_over_noise": float(np.median(ratios)),
            "frac_runs_above_threshold": float((ratios >= TRACK_RATIO_MIN).mean()),
            "median_lag_windows": float(np.median(lags)) if lags.size else None,
            "per_run": track,
        }
    else:
        report["tracking"] = {"n_stepped_scored": 0}

    # ------------------------------------------------------------- VERDICT
    problems = []
    ctrl = sep.get(CONTROL)
    if ctrl is None:
        problems.append(
            f"separation not scored: needs constant-plant runs in all three "
            f"splits, have {sep_n}"
        )
    elif ctrl < CONTROL_MIN:
        problems.append(
            f"POSITIVE CONTROL FAILED: {CONTROL} R2={ctrl:.3f} < {CONTROL_MIN}. "
            "The decode is broken; no other number here is interpretable."
        )
    elif ctrl < SEPARATION_MIN:
        problems.append(f"separation too weak: {CONTROL} R2={ctrl:.3f}")

    tr_rep = report["tracking"]
    if not tr_rep.get("n_stepped_scored"):
        problems.append(
            "no stepped runs scored: this dataset cannot support a tracking "
            "claim (collect with --step-factor)"
        )
    elif tr_rep["median_shift_over_noise"] < TRACK_RATIO_MIN:
        problems.append(
            f"latent does not track: median post-step shift is only "
            f"{tr_rep['median_shift_over_noise']:.2f}x within-segment noise "
            f"(need >= {TRACK_RATIO_MIN})"
        )

    print(f"model: {a.model_dir}")
    print(f"runs: {len(per_run)}   stepped: {len(stepped)}   "
          f"constant per split: {sep_n}")
    if sep:
        print("\nSEPARATION (held-out R2, constant-plant runs)")
        for f in FACTOR_KEYS:
            flag = "  <-- CONTROL" if f == CONTROL else ""
            print(f"  {f:12} {sep[f]:+.3f}{flag}")
    if tr_rep.get("n_stepped_scored"):
        print(f"\nTRACKING ({tr_rep['n_stepped_scored']} stepped runs)")
        print(f"  median shift / noise : {tr_rep['median_shift_over_noise']:.2f}"
              f"   (need >= {TRACK_RATIO_MIN})")
        print(f"  runs above threshold : {tr_rep['frac_runs_above_threshold']:.0%}")
        print(f"  median lag           : {tr_rep['median_lag_windows']} windows"
              f"   (one history window = {a.lag} ticks)")

    print()
    if problems:
        print("GATE FAILED")
        for m in problems:
            print(f"  - {m}")
    else:
        print("GATE PASSED: latent separates plants AND tracks in-episode change")
    report["problems"] = problems
    report["passed"] = not problems

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.out}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
