from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from evaluate_rom import collate_xy, load_checkpoint_model
    from rom_data_contract import DP_SLICE, TARGET_DELTA_DP_SLICE, TARGET_DELTA_EXTENSION_SLICE, VALVE_SLICE
    from rom_dataset import DEFAULT_DATASET_PATH, ROMSequenceDataset, find_default_dataset
except ImportError:
    from kevin_integration.rom_cylinders.evaluate_rom import collate_xy, load_checkpoint_model
    from kevin_integration.rom_cylinders.rom_data_contract import (
        DP_SLICE,
        TARGET_DELTA_DP_SLICE,
        TARGET_DELTA_EXTENSION_SLICE,
        VALVE_SLICE,
    )
    from kevin_integration.rom_cylinders.rom_dataset import DEFAULT_DATASET_PATH, ROMSequenceDataset, find_default_dataset


ROM_DIR = Path(__file__).resolve().parent
DEFAULT_PLOT_DIR = ROM_DIR / "results" / "plots"


def collect_predictions(
    checkpoint: Path,
    dataset: ROMSequenceDataset,
    *,
    batch_size: int,
    max_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model, _ = load_checkpoint_model(checkpoint, dataset, device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_xy)
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    valves: list[np.ndarray] = []
    dps: list[np.ndarray] = []
    seen = 0
    with torch.no_grad():
        for x, y in loader:
            if seen >= max_samples:
                break
            x = x.to(device)
            y = y.to(device)
            pred = dataset.denormalize_y(model(x).detach().cpu()).numpy()
            true = dataset.denormalize_y(y.detach().cpu()).numpy()
            raw_x = dataset.denormalize_x(x.detach().cpu()).numpy()
            take = min(max_samples - seen, pred.shape[0])
            preds.append(pred[:take])
            trues.append(true[:take])
            valves.append(raw_x[:take, -1, VALVE_SLICE])
            dps.append(raw_x[:take, -1, DP_SLICE])
            seen += take
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(valves), np.concatenate(dps)


def save_scatter(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=4, alpha=0.35)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_hist(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=80)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_xy(path: Path, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y_true, s=4, alpha=0.25, label="true")
    ax.scatter(x, y_pred, s=4, alpha=0.25, label="pred")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("DeltaExtension")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_sign_chart(path: Path, true: np.ndarray, pred: np.ndarray, deadband: float) -> None:
    mask = np.abs(true) > deadband
    true_sign = np.sign(true[mask]).astype(int)
    pred_sign = np.sign(pred[mask]).astype(int)
    counts = np.zeros((3, 3), dtype=int)
    mapping = {-1: 0, 0: 1, 1: 2}
    for t, p in zip(true_sign, pred_sign):
        counts[mapping[int(t)], mapping[int(p)]] += 1
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(counts, cmap="Blues")
    ax.set_xticks([0, 1, 2], ["pred -", "pred 0", "pred +"])
    ax.set_yticks([0, 1, 2], ["true -", "true 0", "true +"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training_history(checkpoint: Path, output_dir: Path) -> None:
    history_path = checkpoint.parent / "training_history.json"
    if not history_path.exists():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train"]["loss"] for item in history]
    val_loss = [item["val"]["loss"] for item in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_validation_losses.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot held-out ROM diagnostics.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sign_deadband", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu")
    use_force = bool(payload["config"].get("use_force_input", False))
    dataset_path = args.dataset or (DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else find_default_dataset())
    dataset = ROMSequenceDataset(dataset_path, split=args.split, use_force_input=use_force)
    pred, true, valve, dP = collect_predictions(
        args.checkpoint,
        dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=torch.device(args.device),
    )
    plot_training_history(args.checkpoint, args.output_dir)
    valve_abs = np.abs(valve)
    dP_abs = np.abs(dP)
    for cyl in range(4):
        ext_true = true[:, TARGET_DELTA_EXTENSION_SLICE][:, cyl]
        ext_pred = pred[:, TARGET_DELTA_EXTENSION_SLICE][:, cyl]
        dp_true = true[:, TARGET_DELTA_DP_SLICE][:, cyl]
        dp_pred = pred[:, TARGET_DELTA_DP_SLICE][:, cyl]
        save_scatter(
            args.output_dir / f"cylinder_{cyl + 1}_pred_vs_true_delta_extension.png",
            ext_true,
            ext_pred,
            f"Cylinder {cyl + 1} DeltaExtension",
            "true",
            "predicted",
        )
        save_scatter(
            args.output_dir / f"cylinder_{cyl + 1}_pred_vs_true_delta_dp.png",
            dp_true,
            dp_pred,
            f"Cylinder {cyl + 1} DeltaDP",
            "true",
            "predicted",
        )
        save_hist(
            args.output_dir / f"cylinder_{cyl + 1}_delta_extension_error_hist.png",
            ext_pred - ext_true,
            f"Cylinder {cyl + 1} DeltaExtension Error",
            "prediction error",
        )
        save_xy(
            args.output_dir / f"cylinder_{cyl + 1}_valve_vs_delta_extension.png",
            valve[:, cyl],
            ext_true,
            ext_pred,
            f"Cylinder {cyl + 1} Valve vs DeltaExtension",
            "ValveCmd",
        )
        save_xy(
            args.output_dir / f"cylinder_{cyl + 1}_dp_vs_delta_extension.png",
            dP[:, cyl],
            ext_true,
            ext_pred,
            f"Cylinder {cyl + 1} dP vs DeltaExtension",
            "dP",
        )
        save_sign_chart(
            args.output_dir / f"cylinder_{cyl + 1}_sign_confusion.png",
            ext_true,
            ext_pred,
            args.sign_deadband,
        )
        bins = np.quantile(valve_abs[:, cyl], [0.0, 0.33, 0.66, 1.0])
        bin_rmse = []
        labels = []
        for start, stop in zip(bins[:-1], bins[1:]):
            mask = (valve_abs[:, cyl] >= start) & (valve_abs[:, cyl] <= stop)
            labels.append(f"{start:.2g}-{stop:.2g}")
            bin_rmse.append(float(np.sqrt(np.mean((ext_pred[mask] - ext_true[mask]) ** 2))) if np.any(mask) else np.nan)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, bin_rmse)
        ax.set_title(f"Cylinder {cyl + 1} performance by valve magnitude")
        ax.set_ylabel("DeltaExtension RMSE")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"cylinder_{cyl + 1}_performance_by_valve_magnitude.png", dpi=160)
        plt.close(fig)
    dataset.close()
    print(f"Wrote plots under {args.output_dir}")


if __name__ == "__main__":
    main()
