#!/usr/bin/env python
"""Multi-step (closed-loop) and per-domain evaluation of a Stage 19 SD baseline.

One-step RMSE (teacher-forced, what the trainer reports) is not enough to
claim the SD is a reliable closed-loop plant model: a one-step predictor can
look good while its errors accumulate over a rollout. This tool measures the
actual multi-step behaviour:

    open-loop  per-tick RMSE using the recorded state as input (one-step)
    closed-loop  roll the model forward from a recorded state, feeding its
                 own predicted state back as the next input (the recorded
                 command u_t is used, since actions come from the controller
                 in deployment), and report trajectory RMSE at each horizon
    zero-delta  the persistence baseline (state held constant)

Horizons are in 50 ms ROM ticks (1, 3, 10, 30, 100 -> 0.05..5 s). Rollouts
start at many positions along each run and are averaged. The per-domain
breakdown uses the held-out runs (val + test splits) grouped by domain, so it
shows which deployment dynamics the observable state explains least.

Modes that consume history or a previous-state feature (``su_delta``,
``hist3``, ``hist6``) have a per-run row offset matching the trainer: the
first ``offset`` rows of each run are never evaluated (the trainer drops
them, so evaluating them would feed rows whose history is invented padding).
Rollout starts are likewise restricted to ``start >= offset``.

Usage:
    python evaluate_sd_rollout.py --root datasets/master_v1_frozen --mode hist3
        --checkpoint ../learned_models/sd_model/checkpoints/hist3.pt --out outputs/sd_rollout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AAM_ROOT = REPO / "AAM_module"  # SD tools live in learned_models/sd_model/; aam_data and datasets stay under AAM_module
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_runs_info, load_split  # noqa: E402
from learned_models.sd_model.training.train_sd_baseline import (  # noqa: E402
    CHANNEL_GROUPS,
    SD0MLP,
    input_dim_for,
    load_run_samples,
    unnormalize,
    zscore_apply,
)


HYD_STATE_DIM = 12
CMD_DIM = 8
MECH_CONTEXT_DIM = 10
DEFAULT_HORIZONS = (1, 3, 10, 30, 100)


def row_offset(mode: str) -> int:
    """First valid sample row per run for the mode (trainer drop convention)."""
    if mode == "su_delta":
        return 1
    if mode == "hist3":
        return 2
    if mode == "hist6":
        return 5
    return 0


def group_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for name, (lo, hi) in CHANNEL_GROUPS.items():
        out[name] = float(np.sqrt(np.mean((y_true[:, lo:hi] - y_pred[:, lo:hi]) ** 2)))
    out["all"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return out


def load_run(arrays_root: Path, run_id: int, mode: str) -> dict | None:
    """Load the recorded aligned transitions needed for rollouts.

    Returns s_h (N,12), u (N,8), q (N,10) and the one-step input/target
    matrices for the requested mode (history modes drop the first L-1 ticks).
    """
    sd = arrays_root / "sd" / "dataset"
    path = sd / f"{int(run_id):05d}" / "samples.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        s_h = z["s_h"].astype(np.float64)
        u = z["u"].astype(np.float64)
        q = z["q"].astype(np.float64)
        target = z["target"].astype(np.float64)
    return {"s_h": s_h, "u": u, "q": q, "target": target}


def model_inputs(
    run: dict, mode: str, start: int, length: int, s_roll: np.ndarray,
    state_keep: list[int] | None = None,
) -> np.ndarray:
    """Build the observable input rows for steps start..start+length-1.

    ``s_roll`` holds the (possibly predicted) state at each step; recorded u
    and q are used (actions come from the controller in deployment).
    """
    n = run["s_h"].shape[0]
    s = s_roll[start : start + length]
    if state_keep is not None:
        s = s[:, state_keep]
    u = run["u"][start : start + length]
    if mode == "su":
        return np.concatenate([s, u], axis=-1)
    if mode == "su_delta":
        # [s_t; u_t; s_t - s_{t-1}]; t-1 is the previous (recorded or
        # predicted) state, never an invented row (start >= 1 enforced).
        prev = s_roll[start : start + length] - s_roll[start - 1 : start + length - 1]
        if state_keep is not None:
            prev = prev[:, state_keep]
        return np.concatenate([s, u, prev], axis=-1)
    if mode == "squ":
        q = run["q"][start : start + length]
        return np.concatenate([s, q, u], axis=-1)
    lag = 3 if mode == "hist3" else 6
    rows = []
    for t in range(start, start + length):
        blocks = []
        for k in range(lag):
            t0 = t - k
            if t0 < 0:
                # Warmup rows use the recorded state at tick 0 (no invented
                # padding; matches the trainer's drop-first-lag-1 convention).
                blocks.append(np.concatenate([run["s_h"][0], run["u"][0]]))
            else:
                blocks.append(np.concatenate([s_roll[t0], run["u"][t0]]))
        rows.append(np.concatenate(blocks))
    return np.stack(rows, axis=0)


def rollout_run(
    model, stats, run: dict, mode: str, start: int, horizon: int, device,
    state_keep: list[int] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """One closed-loop rollout; returns (rollout RMSE per group, zero RMSE)."""
    s = run["s_h"].copy()
    preds = []
    for t in range(start, start + horizon):
        X = model_inputs(run, mode, t, 1, s, state_keep)
        x = torch.as_tensor(zscore_apply(X, stats).astype(np.float32), device=device)
        with torch.no_grad():
            delta_n = model(x).detach().cpu().numpy()
        delta = unnormalize(delta_n, stats).reshape(-1)
        preds.append(delta)
        if t + 1 < s.shape[0]:
            s[t + 1] = s[t] + delta
    pred = np.stack(preds, axis=0)  # (horizon, 12) deltas
    true = run["target"][start : start + horizon]
    # Trajectory error: predicted state vs recorded state at each step.
    s0 = run["s_h"][start]
    s_pred = s0 + np.cumsum(pred, axis=0)
    s_true = run["s_h"][start + 1 : start + horizon + 1]
    return group_rmse(s_true, s_pred), group_rmse(s_true, np.broadcast_to(s0[None, :], s_true.shape))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(AAM_ROOT / "datasets/master_v1_frozen"))
    p.add_argument(
        "--mode",
        choices=("su", "su_delta", "squ", "hist3", "hist6"),
        default="hist3",
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--metrics",
        default=None,
        help="metrics.json with the z-score stats matching --checkpoint "
        "(default: learned_models/sd_model/metrics/<mode>.json)",
    )
    p.add_argument(
        "--obs-drop",
        default="",
        help="comma-separated 0-based state channels hidden from the input "
        "(must match the checkpoint's training mask).",
    )
    p.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    p.add_argument("--start-every", type=int, default=50, help="rollout start spacing (ticks)")
    p.add_argument("--max-starts", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(AAM_ROOT / "outputs/sd_rollout.json"))
    args = p.parse_args()

    obs_drop = sorted({int(x) for x in args.obs_drop.split(",") if x.strip()})
    if obs_drop and args.mode in ("hist3", "hist6"):
        raise SystemExit("--obs-drop is not supported for history modes")
    if any(i < 0 or i >= HYD_STATE_DIM for i in obs_drop):
        raise SystemExit(f"--obs-drop indices must be in 0..{HYD_STATE_DIM-1}: {obs_drop}")
    state_keep = [i for i in range(HYD_STATE_DIM) if i not in set(obs_drop)]
    state_keep = state_keep if obs_drop else None

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    root = Path(args.root)
    checkpoint = Path(args.checkpoint or str(REPO / "learned_models/sd_model" / "checkpoints" / f"{args.mode}.pt"))
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    model = SD0MLP(input_dim_for(args.mode, state_keep), hidden=256, depth=3, dropout=0.05).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    metrics_path = (
        Path(args.metrics)
        if args.metrics
        else REPO / "learned_models/sd_model" / "metrics" / f"{args.mode}.json"
    )
    metrics = json.loads(metrics_path.read_text())
    stats = {
        "x_mean": np.asarray(metrics["stats"]["x_mean"], dtype=np.float64),
        "x_std": np.asarray(metrics["stats"]["x_std"], dtype=np.float64),
        "y_mean": np.asarray(metrics["stats"]["y_mean"], dtype=np.float64),
        "y_std": np.asarray(metrics["stats"]["y_std"], dtype=np.float64),
    }

    # Held-out runs (val + test) grouped by domain for the per-domain breakdown.
    val_ids = load_split(root, "val")
    test_ids = load_split(root, "test")
    runs_info = load_runs_info(root)

    def domain_of(run_id: int) -> str:
        return str(runs_info.get(f"{int(run_id):05d}", {}).get("domain", "unknown"))

    eval_ids = sorted(set(val_ids) | set(test_ids))
    per_domain: dict[str, list[int]] = {}
    for rid in eval_ids:
        per_domain.setdefault(domain_of(rid), []).append(rid)

    def evaluate_run(rid: int, rng: np.random.Generator) -> dict:
        run = load_run(root, rid, args.mode)
        if run is None:
            return None
        n = run["s_h"].shape[0]
        offset = row_offset(args.mode)
        # One-step (open-loop) RMSE over the trainer-valid rows of the run.
        X = model_inputs(run, args.mode, offset, n - offset, run["s_h"], state_keep)
        ones = []
        for s0 in range(0, n - offset, 4096):
            x = torch.as_tensor(zscore_apply(X[s0 : s0 + 4096], stats).astype(np.float32), device=device)
            with torch.no_grad():
                ones.append(unnormalize(model(x).detach().cpu().numpy(), stats))
        one_pred = np.concatenate(ones, axis=0)
        one_rmse = group_rmse(run["target"][offset:], one_pred)
        # Rollouts from evenly spaced starts.
        starts = list(range(offset, n - max(args.horizons), args.start_every))
        if args.max_starts and len(starts) > args.max_starts:
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, args.max_starts, dtype=int)]
        roll = {int(h): {k: [] for k in ("extension", "velocity", "dP", "all")} for h in args.horizons}
        zero = {int(h): {k: [] for k in ("extension", "velocity", "dP", "all")} for h in args.horizons}
        for start in starts:
            for h in args.horizons:
                r_rmse, z_rmse = rollout_run(model, stats, run, args.mode, start, int(h), device, state_keep)
                for k in roll[int(h)]:
                    roll[int(h)][k].append(r_rmse[k])
                    zero[int(h)][k].append(z_rmse[k])
        return {
            "one_step_rmse": one_rmse,
            "rollout_rmse": {
                h: {k: float(np.mean(v)) for k, v in groups.items()}
                for h, groups in roll.items()
            },
            "zero_rmse": {
                h: {k: float(np.mean(v)) for k, v in groups.items()}
                for h, groups in zero.items()
            },
            "n_starts": len(starts),
        }

    agg = {
        "one_step_rmse": {k: [] for k in ("extension", "velocity", "dP", "all")},
        "rollout_rmse": {h: {k: [] for k in ("extension", "velocity", "dP", "all")} for h in args.horizons},
        "zero_rmse": {h: {k: [] for k in ("extension", "velocity", "dP", "all")} for h in args.horizons},
    }
    per_domain_report = {}
    for domain, ids in sorted(per_domain.items()):
        d_agg = {
            "one_step_rmse": {k: [] for k in ("extension", "velocity", "dP", "all")},
            "rollout_rmse": {h: {k: [] for k in ("extension", "velocity", "dP", "all")} for h in args.horizons},
            "zero_rmse": {h: {k: [] for k in ("extension", "velocity", "dP", "all")} for h in args.horizons},
        }
        for rid in ids:
            res = evaluate_run(rid, np.random.default_rng(int(rid)))
            if res is None:
                continue
            for k in d_agg["one_step_rmse"]:
                d_agg["one_step_rmse"][k].append(res["one_step_rmse"][k])
                agg["one_step_rmse"][k].append(res["one_step_rmse"][k])
            for h in args.horizons:
                for k in ("extension", "velocity", "dP", "all"):
                    d_agg["rollout_rmse"][h][k].append(res["rollout_rmse"][h][k])
                    d_agg["zero_rmse"][h][k].append(res["zero_rmse"][h][k])
                    agg["rollout_rmse"][h][k].append(res["rollout_rmse"][h][k])
                    agg["zero_rmse"][h][k].append(res["zero_rmse"][h][k])
        per_domain_report[domain] = {
            "n_runs": len(ids),
            "one_step_rmse": {k: float(np.mean(v)) for k, v in d_agg["one_step_rmse"].items()},
            "rollout_rmse": {
                h: {k: float(np.mean(v)) for k, v in groups.items()}
                for h, groups in d_agg["rollout_rmse"].items()
            },
            "zero_rmse": {
                h: {k: float(np.mean(v)) for k, v in groups.items()}
                for h, groups in d_agg["zero_rmse"].items()
            },
        }

    report = {
        "mode": args.mode,
        "checkpoint": str(checkpoint),
        "horizons_ticks": list(args.horizons),
        "start_every_ticks": args.start_every,
        "n_held_out_runs": len(eval_ids),
        "one_step_rmse": {k: float(np.mean(v)) for k, v in agg["one_step_rmse"].items()},
        "rollout_rmse": {
            h: {k: float(np.mean(v)) for k, v in groups.items()}
            for h, groups in agg["rollout_rmse"].items()
        },
        "zero_rmse": {
            h: {k: float(np.mean(v)) for k, v in groups.items()}
            for h, groups in agg["zero_rmse"].items()
        },
        "per_domain": per_domain_report,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"mode={args.mode} held-out runs={eval_ids}")
    print("one-step RMSE:", {k: f"{v:.4g}" for k, v in report["one_step_rmse"].items()})
    print(f"{'H(ticks)':>9} {'roll_ext':>11} {'roll_vel':>11} {'roll_dP':>11} {'zero_all':>11}")
    for h in args.horizons:
        r = report["rollout_rmse"][h]
        print(
            f"{h:>9} {r['extension']:>11.4g} {r['velocity']:>11.4g} "
            f"{r['dP']:>11.4g} {report['zero_rmse'][h]['all']:>11.4g}"
        )
    print("per-domain (held-out):")
    for domain, d in per_domain_report.items():
        r30 = d["rollout_rmse"].get(30)
        r30_all = r30["all"] if r30 is not None else float("nan")
        print(
            f"  {domain:<11} runs={d['n_runs']} "
            f"one_all={d['one_step_rmse']['all']:.4g} roll30_all={r30_all:.4g}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
