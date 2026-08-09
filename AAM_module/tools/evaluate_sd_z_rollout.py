#!/usr/bin/env python
"""Closed-loop rollout evaluation for the Stage 20-22 privileged SD_z model.

Same protocol as ``tools/evaluate_sd_rollout.py`` (one-step + free-running
multi-step with the model's own state fed back, recorded commands) but for
``SD_z(s_t, u_t, q_t, h_t, z_i)``: the latent ``z_i`` is computed once per
run from the run's true factor vector with the TRAIN-ONLY normalization
saved in the trainer's ``metrics.json``, then held fixed for the whole
rollout (stationary configuration per run). When the checkpoint was trained
with the Stage 22 v3 causal-belief protocol, the GRU belief ``h_t`` is
initialized from the causal window at each rollout start and carried
forward with the predicted state (RMA/PlaNet-style dynamic belief), while
``z`` stays the static configuration latent.

``--latent-mode`` implements the Stage 22 latent-matters check. Multiple
modes can be requested in one call so the comparisons are PAIRED per run:

- ``true``: z_i = mu_psi(e_tilde_i), the privileged factor latent.
- ``zero``: z = 0 for every run (SD_z(s, u, 0)).
- ``shuffled``: each held-out run receives a factor vector sampled without
  replacement from the TRAINING runs (fixed seed 22). This destroys pairing
  without using held-out factors to construct the control.
- ``random``: each held-out run receives a random TRAINING factor vector
  with replacement (seed 23).
- ``mean``: every run receives the mean TRAINING latent. This is the clean
  fixed-context control; no held-out/test factors enter its construction.
- ``nominal``: every run receives the encoding of the nominal physical
  configuration (flow/compliance/seal/pressure/com_z offsets zero,
  mass_scale one).

Report: per-mode mean rollout RMSE, per-run tables, PAIRED mean differences
``true - control`` per channel group at every horizon with percentile
bootstrap 95% CIs over runs (pooled and separately for val/test), and the
val/test split breakdown so the five nominal test runs are reported (and
judged) separately from the adaptive validation pool. ``--q-rollout`` makes
the mechanical-context contract explicit: ``recorded`` is a conditional
hydraulic forecast with future measured q, while ``hold-start`` is the
leakage-free open-loop sensitivity check available without a learned
mechanical rollout model.

Usage:
    python evaluate_sd_z_rollout.py --root datasets/master_v1_frozen \
        --model-dir outputs/sd_z/dz4 \
        --latent-mode true zero shuffled random mean nominal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_runs_info, load_split  # noqa: E402
from tools.train_sd_z import (  # noqa: E402
    FACTOR_KEYS,
    BeliefEncoder,
    FactorEncoder,
    SDz,
    causal_windows_np,
    group_rmse,
)


DEFAULT_HORIZONS = (1, 3, 10, 30, 100)
HYD_STATE_DIM = 12
CMD_DIM = 8
CHANNELS = ("extension", "velocity", "dP", "all")
NOMINAL_FACTORS = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
SHUFFLE_SEED = 22
RANDOM_SEED = 23
CI_SEED = 7
CI_BOOTSTRAPS = 2000


def load_run(sd_root: Path, rid: int) -> dict | None:
    path = sd_root / f"{int(rid):05d}" / "samples.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads((sd_root / f"{int(rid):05d}" / "sample_meta.json").read_text())
        return {
            "s": z["s_h"].astype(np.float64),
            "u": z["u"].astype(np.float64),
            "q": z["q"].astype(np.float64),
            "target": z["target"].astype(np.float64),
            "e": np.asarray([float(meta["z"][k]) for k in FACTOR_KEYS], dtype=np.float64),
        }


def bootstrap_ci(diffs: np.ndarray, seed: int = CI_SEED,
                 n: int = CI_BOOTSTRAPS, alpha: float = 0.05) -> list[float] | None:
    """Percentile bootstrap 95% CI of the mean of per-run paired differences."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, d.size, size=d.size)
        means[i] = d[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(AAM_ROOT / "datasets/master_v1_frozen"))
    p.add_argument("--model-dir", required=True, help="train_sd_z.py output dir")
    p.add_argument(
        "--latent-mode",
        choices=("true", "zero", "shuffled", "random", "mean", "nominal"),
        nargs="+",
        default=["true"],
        help="which latents the SD_z rollout consumes; multiple modes are "
        "evaluated in one call so per-run paired comparisons are valid",
    )
    p.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    p.add_argument("--start-every", type=int, default=50)
    p.add_argument("--max-starts", type=int, default=200)
    p.add_argument(
        "--q-rollout",
        choices=("recorded", "hold-start"),
        default="recorded",
        help="future joint-context protocol for multi-step rollout: "
        "'recorded' conditions on the measured q at every future tick; "
        "'hold-start' uses q at the rollout start for the whole horizon",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    model_dir = Path(args.model_dir)
    metrics = json.loads((model_dir / "metrics.json").read_text())
    latent_source = metrics.get("latent_source", "encoder")
    dz = int(metrics["latent_dim"])
    if latent_source == "none":
        latent_in = 0
    elif latent_source == "encoder":
        latent_in = int(metrics.get("latent_in_dim", dz))
    else:
        latent_in = int(metrics.get("latent_in_dim", 6))
    state_keep = [int(i) for i in metrics.get("state_keep", list(range(HYD_STATE_DIM)))]
    state_dim = len(state_keep)
    use_q = bool(metrics.get("use_q", False))
    use_belief = bool(metrics.get("use_belief", False))
    belief_dim = int(metrics.get("belief_dim", 0))
    history_lag = int(metrics.get("history_lag", 0))
    training_q_mode = str(metrics.get("rollout_q_mode", "recorded"))

    encoder = FactorEncoder(dz).to(device) if latent_source == "encoder" else None
    belief = (
        BeliefEncoder(state_dim + CMD_DIM, belief_dim).to(device)
        if use_belief
        else None
    )
    sd = SDz(latent_in, state_dim=state_dim, hidden=256, depth=3, dropout=0.05,
             use_q=use_q, use_belief=use_belief, belief_dim=belief_dim).to(device)
    if encoder is not None:
        encoder.load_state_dict(torch.load(model_dir / "encoder.pt", map_location=device))
    if belief is not None:
        belief.load_state_dict(torch.load(model_dir / "belief.pt", map_location=device))
    sd.load_state_dict(torch.load(model_dir / "sd.pt", map_location=device))
    if encoder is not None:
        encoder.eval()
    if belief is not None:
        belief.eval()
    sd.eval()

    e_mean = np.asarray(metrics["e_normalization"]["mean"], dtype=np.float64)
    e_std = np.asarray(metrics["e_normalization"]["std"], dtype=np.float64)
    s_mean = np.asarray(metrics["s_stats"]["mean"], dtype=np.float64)
    s_std = np.asarray(metrics["s_stats"]["std"], dtype=np.float64)
    u_mean = np.asarray(metrics["u_stats"]["mean"], dtype=np.float64)
    u_std = np.asarray(metrics["u_stats"]["std"], dtype=np.float64)
    y_mean = np.asarray(metrics["y_stats"]["mean"], dtype=np.float64)
    y_std = np.asarray(metrics["y_stats"]["std"], dtype=np.float64)
    q_mean = np.asarray(metrics.get("q_stats", {}).get("mean",
                       np.zeros(10, dtype=np.float64).tolist()), dtype=np.float64)
    q_std = np.asarray(metrics.get("q_stats", {}).get("std",
                       np.ones(10, dtype=np.float64).tolist()), dtype=np.float64)

    s_mean_t = torch.as_tensor(s_mean, dtype=torch.float32, device=device)
    s_std_t = torch.as_tensor(s_std, dtype=torch.float32, device=device)
    u_mean_t = torch.as_tensor(u_mean, dtype=torch.float32, device=device)
    u_std_t = torch.as_tensor(u_std, dtype=torch.float32, device=device)
    q_mean_t = torch.as_tensor(q_mean, dtype=torch.float32, device=device)
    q_std_t = torch.as_tensor(q_std, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)

    root = Path(args.root)
    sd_root = root / "sd" / "dataset"
    runs_info = load_runs_info(root)
    train_ids = sorted(load_split(root, "train"))
    val_ids = sorted(load_split(root, "val"))
    test_ids = sorted(load_split(root, "test"))
    held = sorted(set(val_ids) | set(test_ids))
    split_of = {int(rid): ("val" if int(rid) in val_ids else "test") for rid in held}

    # Precompute the per-run latent for each requested mode.
    @torch.no_grad()
    def encode(e: np.ndarray) -> torch.Tensor:
        e_n = torch.as_tensor((e - e_mean) / e_std, dtype=torch.float32, device=device)
        return e_n[None, :] if encoder is None else encoder(e_n[None, :])

    train_runs = {
        int(rid): run
        for rid in train_ids
        if (run := load_run(sd_root, rid)) is not None
    }
    if not train_runs:
        raise SystemExit("no training runs available for leakage-free controls")
    train_pool = sorted(train_runs)

    donor_map: dict[str, dict[int, int]] = {}
    if "shuffled" in args.latent_mode:
        rng = np.random.default_rng(SHUFFLE_SEED)
        # Sampling without replacement retains empirical training support and
        # avoids constructing a test control from other held-out factors.
        donors = rng.choice(train_pool, size=len(held), replace=len(train_pool) < len(held))
        donor_map["shuffled"] = {int(held[i]): int(donors[i]) for i in range(len(held))}
    if "random" in args.latent_mode:
        rng = np.random.default_rng(RANDOM_SEED)
        donor_map["random"] = {
            int(rid): int(train_pool[int(rng.integers(0, len(train_pool)))])
            for rid in held
        }

    mean_z = None
    if "mean" in args.latent_mode:
        mean_z = torch.mean(
            torch.cat([encode(train_runs[rid]["e"]) for rid in train_pool], dim=0),
            dim=0,
            keepdim=True,
        )
    nominal_z = encode(NOMINAL_FACTORS) if "nominal" in args.latent_mode else None

    def z_for(mode: str, rid: int, run: dict) -> torch.Tensor:
        if mode == "zero" or latent_source == "none":
            return torch.zeros(1, latent_in, dtype=torch.float32, device=device)
        if mode == "mean":
            return mean_z
        if mode == "nominal":
            return nominal_z
        if mode in ("shuffled", "random"):
            donor = donor_map[mode][int(rid)]
            return encode(train_runs[donor]["e"])
        return encode(run["e"])

    def step(s_keep: torch.Tensor, u: torch.Tensor, q: torch.Tensor,
             h: torch.Tensor | None, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One SD step: belief update, delta prediction (raw units)."""
        s_n = (s_keep - s_mean_t) / s_std_t
        u_n = (u - u_mean_t) / u_std_t
        if use_belief:
            h = belief.step(torch.cat([s_n, u_n], dim=-1), h)
        cols = [s_n, u_n]
        if use_q:
            cols.append((q - q_mean_t) / q_std_t)
        if use_belief:
            cols.append(h)
        X = torch.cat(cols + [z.expand(s_keep.shape[0], -1)], dim=-1)
        return sd(X) * y_std_t + y_mean_t, h

    def init_belief(s_keep: np.ndarray, u: np.ndarray, start: int) -> torch.Tensor | None:
        if not use_belief:
            return None
        blocks = causal_windows_np(
            s_keep[start - history_lag + 1 : start + 1],
            u[start - history_lag + 1 : start + 1],
            history_lag,
        )
        bn = torch.as_tensor(
            np.concatenate(
                [(blocks[0, :, :state_dim] - s_mean) / s_std,
                 (blocks[0, :, state_dim:] - u_mean) / u_std],
                axis=-1,
            ).astype(np.float32)[None, :, :],
            device=device,
        )
        return belief.init_from_window(bn)

    offset = history_lag - 1 if use_belief else 0

    # Per-mode per-run metrics, collected in one pass over the held-out runs.
    per_mode_run: dict[str, dict[int, dict]] = {
        m: {} for m in args.latent_mode
    }
    per_mode_split: dict[str, dict[str, dict]] = {
        m: {"val": {h: {c: [] for c in CHANNELS} for h in args.horizons},
            "test": {h: {c: [] for c in CHANNELS} for h in args.horizons}}
        for m in args.latent_mode
    }

    for rid in held:
        run = load_run(sd_root, rid)
        if run is None:
            continue
        n = run["s"].shape[0]
        if n <= offset:
            continue
        s = torch.as_tensor(run["s"], dtype=torch.float32, device=device)
        u = torch.as_tensor(run["u"], dtype=torch.float32, device=device)
        q = torch.as_tensor(run["q"], dtype=torch.float32, device=device)
        split = split_of[int(rid)]

        for mode in args.latent_mode:
            z = z_for(mode, rid, run)
            # One-step (open loop) over valid rows, belief per-row windows.
            one_preds = []
            B = n - offset
            if use_belief:
                blocks = causal_windows_np(run["s"][:, state_keep], run["u"], history_lag)
                bn = torch.as_tensor(
                    np.concatenate(
                        [(blocks[:, :, :state_dim] - s_mean) / s_std,
                         (blocks[:, :, state_dim:] - u_mean) / u_std],
                        axis=-1,
                    ).astype(np.float32),
                    device=device,
                )
                h_all = belief.init_from_window(bn)
            else:
                h_all = None
            for start in range(0, B, 8192):
                b = slice(start, start + 8192)
                s_keep = s[offset:][b][:, state_keep]
                ub = u[offset:][b]
                qb = q[offset:][b] if use_q else u[offset:][b][:, :0]
                hb = h_all[b] if h_all is not None else None
                d, _ = step(s_keep, ub, qb, hb, z)
                one_preds.append(d.detach().cpu().numpy())
            one_pred = np.concatenate(one_preds, axis=0)
            one_rmse = group_rmse(run["target"][offset:], one_pred)

            starts = list(range(offset, n - max(args.horizons), args.start_every))
            if args.max_starts and len(starts) > args.max_starts:
                starts = [starts[i] for i in np.linspace(0, len(starts) - 1, args.max_starts, dtype=int)]
            run_h = {h: {c: [] for c in CHANNELS} for h in args.horizons}
            for start in starts:
                s_pred = s[start].clone()
                h = init_belief(run["s"][:, state_keep], run["u"], start)
                preds_h: dict[int, list[torch.Tensor]] = {h_lim: [] for h_lim in args.horizons}
                for t in range(start, start + max(args.horizons)):
                    q_tick = q[t : t + 1] if args.q_rollout == "recorded" else q[start : start + 1]
                    d, h = step(s_pred[None, state_keep], u[t : t + 1],
                                q_tick if use_q else u[t : t + 1, :0], h, z)
                    for h_lim in args.horizons:
                        if t - start < h_lim:
                            preds_h[h_lim].append(d[0])
                    s_pred = s_pred + d[0]
                for h_lim in args.horizons:
                    pred_arr = torch.stack(preds_h[h_lim], dim=0).detach().cpu().numpy()
                    s_true = run["s"][start + 1 : start + 1 + h_lim]
                    s_pred_arr = run["s"][start] + np.cumsum(pred_arr, axis=0)
                    rm = group_rmse(s_true, s_pred_arr)
                    for c in CHANNELS:
                        run_h[h_lim][c].append(rm[c])

            per_mode_run[mode][int(rid)] = {
                "split": split,
                "one_step_rmse": {c: one_rmse[c] for c in CHANNELS},
                "rollout_rmse": {h_lim: {c: float(np.mean(run_h[h_lim][c])) for c in CHANNELS}
                                 for h_lim in args.horizons},
            }
            for h_lim in args.horizons:
                for c in CHANNELS:
                    per_mode_split[mode][split][h_lim][c].append(
                        per_mode_run[mode][int(rid)]["rollout_rmse"][h_lim][c]
                    )

    # Aggregate + paired comparisons.
    agg = {m: {h: {c: float(np.mean([per_mode_run[m][r]["rollout_rmse"][h][c]
                                        for r in per_mode_run[m]]))
                    for c in CHANNELS} for h in args.horizons}
            for m in args.latent_mode}
    # Persistence baseline (state held constant) per horizon, pooled over runs.
    zero_agg: dict[int, dict[str, float]] = {}
    for h in args.horizons:
        acc = {c: [] for c in CHANNELS}
        for rid in held:
            run = load_run(sd_root, rid)
            if run is None or run["s"].shape[0] <= offset:
                continue
            starts = list(range(offset, run["s"].shape[0] - h, args.start_every))
            for start in starts:
                s_true = run["s"][start + 1 : start + 1 + h]
                rm = group_rmse(s_true, np.broadcast_to(run["s"][start][None, :], s_true.shape))
                for c in CHANNELS:
                    acc[c].append(rm[c])
        zero_agg[h] = {c: float(np.mean(v)) for c, v in acc.items()}
    paired: dict[str, dict] = {}
    if "true" in args.latent_mode:
        common = sorted(set(per_mode_run["true"]))
        for mode in args.latent_mode:
            if mode == "true":
                continue
            paired[mode] = {}
            for h in args.horizons:
                paired[mode][h] = {}
                for c in CHANNELS:
                    diffs = np.asarray(
                        [per_mode_run["true"][r]["rollout_rmse"][h][c]
                         - per_mode_run[mode][r]["rollout_rmse"][h][c]
                         for r in common if r in per_mode_run[mode]],
                        dtype=np.float64,
                    )
                    paired[mode][h][c] = {
                        "mean_diff_true_minus_control": float(diffs.mean()) if diffs.size else None,
                        "bootstrap_95ci": bootstrap_ci(diffs),
                        "n_runs": int(diffs.size),
                    }

    paired_by_split: dict[str, dict[str, dict]] = {"val": {}, "test": {}}
    if "true" in args.latent_mode:
        for split in ("val", "test"):
            split_runs = [r for r in common if split_of[int(r)] == split]
            for mode in args.latent_mode:
                if mode == "true":
                    continue
                paired_by_split[split][mode] = {}
                for h in args.horizons:
                    paired_by_split[split][mode][h] = {}
                    for c in CHANNELS:
                        diffs = np.asarray(
                            [per_mode_run["true"][r]["rollout_rmse"][h][c]
                             - per_mode_run[mode][r]["rollout_rmse"][h][c]
                             for r in split_runs if r in per_mode_run[mode]],
                            dtype=np.float64,
                        )
                        paired_by_split[split][mode][h][c] = {
                            "mean_diff_true_minus_control": (
                                float(diffs.mean()) if diffs.size else None
                            ),
                            "bootstrap_95ci": bootstrap_ci(diffs),
                            "n_runs": int(diffs.size),
                        }

    report = {
        "model_dir": str(model_dir),
        "latent_dim": dz,
        "latent_source": latent_source,
        "use_q": use_q,
        "use_belief": use_belief,
        "belief_dim": belief_dim,
        "history_lag": history_lag,
        "obs_drop": metrics.get("obs_drop", []),
        "q_rollout": args.q_rollout,
        "training_q_mode": training_q_mode,
        "control_construction": {
            "donor_pool": "training runs only",
            "mean": "mean encoded latent over training runs",
            "nominal_factors": NOMINAL_FACTORS.tolist(),
            "shuffle_seed": SHUFFLE_SEED,
            "random_seed": RANDOM_SEED,
        },
        "horizons_ticks": list(args.horizons),
        "modes": list(args.latent_mode),
        "eval_pool": {"val": val_ids, "test": test_ids},
        "rollout_rmse": agg,
        "zero_rmse": zero_agg,
        "paired_vs_true": paired,
        "paired_vs_true_by_split": paired_by_split,
        "per_run": per_mode_run,
        "splits": {
            sp: {
                m: {h: {c: float(np.mean(v)) for c, v in per_mode_split[m][sp][h].items()}
                    for h in args.horizons}
                for m in args.latent_mode
            }
            for sp in ("val", "test")
        },
    }
    out = Path(args.out) if args.out else model_dir / "rollout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"latent_source={latent_source} use_belief={use_belief} lag={history_lag} "
          f"modes={list(args.latent_mode)}")
    print(f"{'H':>4}" + "".join(f" {m[:8]:>12}" for m in args.latent_mode) + " (rollout all)")
    for h in args.horizons:
        print(f"{h:>4}" + "".join(f" {agg[m][h]['all']:>12.4g}" for m in args.latent_mode))
    if paired:
        print(
            f"paired mean diff (true - control), H{args.horizons[-1]}: "
            "negative = true better"
        )
        for mode, table in paired.items():
            row = []
            for c in ("extension", "velocity", "dP", "all"):
                v = table[args.horizons[-1]][c]
                row.append(f"{c[:3]}={v['mean_diff_true_minus_control']:.4g}"
                           f"({v['bootstrap_95ci'][0]:.3g},{v['bootstrap_95ci'][1]:.3g})" if v["bootstrap_95ci"] else f"{c[:3]}=NA")
            print(f"  {mode:>10}: " + " ".join(row))
    print(f"splits (H{args.horizons[-1]} all):", {
        sp: {m: f"{report['splits'][sp][m][args.horizons[-1]]['all']:.4g}" for m in args.latent_mode}
        for sp in ("val", "test")
    })
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
