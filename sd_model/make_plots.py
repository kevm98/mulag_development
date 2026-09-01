#!/usr/bin/env python
"""Generate the Stage 19 SD baseline training plots inside ``sd_model/``.

Reads the per-mode metrics (``metrics/*.json``), the stored test-set parity
predictions (``data/parity_<mode>.npz``) and the zero-delta reference
(``data/zero_baseline.json``) and writes:

    plots/loss_curves.png          train/val loss vs epoch, all modes
    plots/val_rmse_curves.png      val RMSE per channel group vs epoch
    plots/test_rmse_comparison.png test RMSE bars per mode + zero baseline
    plots/r2_comparison.png        R2 vs the zero-delta predictor per mode
    plots/parity_hist3.png         predicted vs target scatter (best mode)
    plots/rollout_rmse_vs_horizon.png  closed-loop trajectory error growth
    plots/per_domain_rmse.png          held-out error per deployment domain
    plots/summary_table.csv        RMSE/R2 table

If the frozen dataset is still available (``--root``, default
``/tmp/stage19_frozen``) the parity data and zero baseline are recomputed
from the test runs instead of read from ``data/``, so the plots stay
reproducible from either source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "AAM_module") not in sys.path:
    sys.path.insert(0, str(REPO / "AAM_module"))

from sd_model.training.train_sd_baseline import (  # noqa: E402
    CHANNEL_GROUPS,
    SD0MLP,
    TARGET_DIM,
    input_dim_for,
    load_run_samples,
    unnormalize,
    zscore_apply,
)


MODES = ("su", "su_delta", "squ", "hist3", "hist6")
MODE_LABELS = {
    "su": "SD0(s,u) 20-D",
    "su_delta": "SD0 + delta 32-D",
    "squ": "SD0(s,q,u) 30-D",
    "hist3": "SD0 + hist L=3 60-D",
    "hist6": "SD0 + hist L=6 120-D",
}
GROUP_LABELS = {"extension": "extension [m]", "velocity": "velocity [m/s]", "dP": "dP [Pa]"}
COLORS = {
    "su": "#1f77b4",
    "su_delta": "#ff7f0e",
    "squ": "#2ca02c",
    "hist3": "#d62728",
    "hist6": "#9467bd",
}


def load_metrics(model_dir: Path, mode: str) -> dict:
    return json.loads((model_dir / "metrics" / f"{mode}.json").read_text())


def zero_baseline_from_data(root: Path, test_ids: list[int]) -> dict:
    ys = []
    sd = root / "sd" / "dataset"
    for rid in test_ids:
        data = load_run_samples(sd / f"{int(rid):05d}", "su")
        if data is not None:
            ys.append(data[1])
    y = np.concatenate(ys, axis=0)
    out = {}
    for name, (lo, hi) in CHANNEL_GROUPS.items():
        out[name] = float(np.sqrt(np.mean(y[:, lo:hi] ** 2)))
    out["all"] = float(np.sqrt(np.mean(y**2)))
    return out


def predict_test(root: Path, test_ids: list[int], mode: str, device: str) -> dict:
    """Best-checkpoint one-step predictions on the test runs (physical units)."""
    import torch

    dev = torch.device(device)
    model = SD0MLP(input_dim_for(mode), hidden=256, depth=3, dropout=0.05).to(dev)
    model.load_state_dict(torch.load(REPO / "sd_model" / "checkpoints" / f"{mode}.pt", map_location=dev))
    model.eval()
    metrics = load_metrics(REPO / "sd_model", mode)
    stats = {
        "x_mean": np.asarray(metrics["stats"]["x_mean"], dtype=np.float64),
        "x_std": np.asarray(metrics["stats"]["x_std"], dtype=np.float64),
        "y_mean": np.asarray(metrics["stats"]["y_mean"], dtype=np.float64),
        "y_std": np.asarray(metrics["stats"]["y_std"], dtype=np.float64),
    }
    xs, ys = [], []
    sd = root / "sd" / "dataset"
    for rid in test_ids:
        data = load_run_samples(sd / f"{int(rid):05d}", mode)
        if data is None:
            continue
        x, y = data
        xs.append(zscore_apply(x, stats).astype(np.float32))
        ys.append(y)
    if not xs:
        raise SystemExit(f"no test data for mode {mode}")
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], 8192):
            xb = torch.as_tensor(X[start : start + 8192], device=dev)
            preds.append(model(xb).detach().cpu().numpy())
    y_pred = unnormalize(np.concatenate(preds, axis=0), stats)
    return {"target": y.astype(np.float64), "pred": y_pred.astype(np.float64)}


def r2_vs_zero(test_rmse_all: float, zero_all: float) -> float:
    return float(1.0 - (test_rmse_all / zero_all) ** 2)


def plot_loss_curves(history: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for mode, hist in history.items():
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], label=MODE_LABELS[mode], color=COLORS[mode])
        axes[1].plot(ep, [h["val_loss"] for h in hist], label=MODE_LABELS[mode], color=COLORS[mode])
    for ax, title in zip(axes, ("train loss (normalized)", "val loss (normalized)")):
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Stage 19 SD0 baseline: loss curves")
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=140)
    plt.close(fig)


def plot_val_rmse_curves(history: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for mode, hist in history.items():
        ep = [h["epoch"] for h in hist]
        for ax, group in zip(axes, CHANNEL_GROUPS):
            vals = [h.get("val_rmse", {}).get(group) for h in hist]
            ax.plot(ep, vals, label=MODE_LABELS[mode], color=COLORS[mode])
            ax.set_yscale("log")
            ax.set_title(f"val RMSE {GROUP_LABELS[group]}")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Stage 19 SD0 baseline: validation RMSE per channel group")
    fig.tight_layout()
    fig.savefig(out / "val_rmse_curves.png", dpi=140)
    plt.close(fig)


def plot_test_rmse_comparison(summary: dict, zero: dict, out: Path) -> None:
    groups = ("extension", "velocity", "dP")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(MODES))
    for ax, group in zip(axes, groups):
        vals = [summary[m]["test_rmse"][group] for m in MODES]
        bars = ax.bar(x, vals, color=[COLORS[m] for m in MODES], width=0.62)
        ax.axhline(zero[group], color="black", ls="--", lw=1.2, label="zero-delta")
        ax.set_xticks(x, [MODE_LABELS[m] for m in MODES], rotation=20, fontsize=7)
        ax.set_title(f"test RMSE {GROUP_LABELS[group]}")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2g}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Stage 19 SD0 baseline: held-out test error vs zero-delta")
    fig.tight_layout()
    fig.savefig(out / "test_rmse_comparison.png", dpi=140)
    plt.close(fig)


def plot_r2_comparison(summary: dict, zero_all: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    r2 = [summary[m]["r2_vs_zero"] for m in MODES]
    bars = ax.bar(np.arange(len(MODES)), r2, color=[COLORS[m] for m in MODES], width=0.6)
    ax.set_xticks(np.arange(len(MODES)), [MODE_LABELS[m] for m in MODES], rotation=15, fontsize=8)
    ax.set_ylabel("R2 vs zero-delta (test)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Stage 19 SD0 baseline: explained one-step variance on unseen config")
    for b, v in zip(bars, r2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "r2_comparison.png", dpi=140)
    plt.close(fig)


def plot_parity(mode: str, target: np.ndarray, pred: np.ndarray, out: Path) -> None:
    n = target.shape[0]
    rng = np.random.RandomState(0)
    idx = rng.choice(n, size=min(n, 4000), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (group, (lo, hi)) in zip(axes, CHANNEL_GROUPS.items()):
        t = target[idx, lo:hi].reshape(-1)
        p = pred[idx, lo:hi].reshape(-1)
        ax.scatter(t, p, s=3, alpha=0.25, color=COLORS[mode])
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(f"target {GROUP_LABELS[group]}")
        ax.set_ylabel(f"pred {GROUP_LABELS[group]}")
        ax.set_title(group)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Stage 19 SD0 baseline: one-step parity on held-out test ({MODE_LABELS[mode]})")
    fig.tight_layout()
    fig.savefig(out / f"parity_{mode}.png", dpi=140)
    plt.close(fig)


def plot_rollout(rollout: dict[str, dict], out: Path) -> None:
    """Closed-loop trajectory RMSE vs horizon (all channels), vs zero-delta."""
    modes = [m for m in MODES if m in rollout]
    horizons = sorted({int(h) for m in modes for h in rollout[m]["rollout_rmse"]})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for mode in modes:
        rr = rollout[mode]["rollout_rmse"]
        ax.plot(
            horizons,
            [rr[str(h)]["all"] for h in horizons],
            marker="o",
            label=MODE_LABELS[mode],
            color=COLORS[mode],
        )
    if modes:
        zero = rollout[modes[0]]["zero_rmse"]
        ax.plot(
            horizons,
            [zero[str(h)]["all"] for h in horizons],
            "k--",
            lw=1.4,
            label="zero-delta (persistence)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rollout horizon [50 ms ticks]")
    ax.set_ylabel("trajectory RMSE (all channels)")
    ax.set_title("Stage 19 SD0 baseline: closed-loop rollout error growth")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rollout_rmse_vs_horizon.png", dpi=140)
    plt.close(fig)


def plot_per_domain(rollout: dict[str, dict], out: Path) -> None:
    """Held-out one-step RMSE per deployment domain (best mode with data)."""
    candidates = [m for m in ("hist3", "su") if m in rollout and rollout[m].get("per_domain")]
    if not candidates:
        return
    mode = candidates[0]
    per_domain = rollout[mode]["per_domain"]
    domains = sorted(per_domain)
    one = [per_domain[d]["one_step_rmse"]["all"] for d in domains]
    labels = [f"{d}\n(n={per_domain[d]['n_runs']})" for d in domains]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(np.arange(len(domains)), one, color="#d62728", width=0.6)
    ax.set_xticks(np.arange(len(domains)), labels, fontsize=9)
    ax.set_ylabel("held-out one-step RMSE (all channels)")
    ax.set_title(f"Stage 19 SD0 baseline: which dynamics are explained least ({MODE_LABELS[mode]})")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, one):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "per_domain_rmse.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/tmp/stage19_frozen", help="frozen dataset (optional)")
    p.add_argument("--device", default="cuda" if _cuda() else "cpu")
    p.add_argument("--model-dir", default=str(REPO / "sd_model"))
    p.add_argument("--force-recompute", action="store_true")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = model_dir / "plots"
    data_dir = model_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    have_dataset = (root / "splits" / "test.json").is_file()

    history = {m: load_metrics(model_dir, m)["history"] for m in MODES}
    summary = {}
    for m in MODES:
        metrics = load_metrics(model_dir, m)
        summary[m] = {"test_rmse": metrics["test_rmse"], "val_rmse": metrics["val_rmse"]}

    # Zero-delta reference (data property; recompute when the dataset is present).
    zero_path = data_dir / "zero_baseline.json"
    if have_dataset and args.force_recompute:
        from aam_data import load_split

        zero = zero_baseline_from_data(root, load_split(root, "test"))
        zero_path.write_text(json.dumps(zero, indent=2) + "\n")
    elif zero_path.is_file():
        zero = json.loads(zero_path.read_text())
    elif have_dataset:
        from aam_data import load_split

        zero = zero_baseline_from_data(root, load_split(root, "test"))
        zero_path.write_text(json.dumps(zero, indent=2) + "\n")
    else:
        raise SystemExit("no zero-baseline data and no dataset; run once with --root")

    for m in MODES:
        summary[m]["r2_vs_zero"] = r2_vs_zero(summary[m]["test_rmse"]["all"], zero["all"])

    plot_loss_curves(history, out_dir)
    plot_val_rmse_curves(history, out_dir)
    plot_test_rmse_comparison(summary, zero, out_dir)
    plot_r2_comparison(summary, zero["all"], out_dir)

    # Optional: rollout + per-domain plots from data/rollout_<mode>.json files
    # produced by sd_model/evaluation/evaluate_sd_rollout.py.
    rollout = {}
    for m in MODES:
        rpath = data_dir / f"rollout_{m}.json"
        if rpath.is_file():
            rollout[m] = json.loads(rpath.read_text())
    if rollout:
        plot_rollout(rollout, out_dir)
        plot_per_domain(rollout, out_dir)

    # Parity data: prefer the stored npz; recompute from the dataset when asked
    # or when the stored file is missing.
    parity_mode = "hist3"
    parity_path = data_dir / f"parity_{parity_mode}.npz"
    if have_dataset and (args.force_recompute or not parity_path.is_file()):
        from aam_data import load_split

        parity = predict_test(root, load_split(root, "test"), parity_mode, args.device)
        np.savez_compressed(parity_path, target=parity["target"], pred=parity["pred"])
        print(f"recomputed parity from dataset -> {parity_path}")
    if parity_path.is_file():
        with np.load(parity_path, allow_pickle=False) as z:
            plot_parity(parity_mode, z["target"], z["pred"], out_dir)

    # Summary table.
    rows = []
    for m in MODES:
        tr = summary[m]["test_rmse"]
        rows.append(
            {
                "mode": m,
                "input_dim": input_dim_for(m),
                "test_rmse_extension_m": tr["extension"],
                "test_rmse_velocity_mps": tr["velocity"],
                "test_rmse_dP_Pa": tr["dP"],
                "test_rmse_all": tr["all"],
                "r2_vs_zero": summary[m]["r2_vs_zero"],
            }
        )
    import csv

    with (out_dir / "summary_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"plots -> {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print("  ", f.name)


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
