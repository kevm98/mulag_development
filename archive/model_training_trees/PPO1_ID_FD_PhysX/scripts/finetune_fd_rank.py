#!/usr/bin/env python3
"""Fine-tune the cylinder FD model without collapsing valve-command rank.

The model and its normalization contract come from the original checkpoint.  The
rank target is the normalized-space valve partial effect in the original v7.3
MATLAB data, after linearly regressing the other 16 inputs out of both valve and
target.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "PPO1_ID_FD_PhysX/models/fd/FD_cylinder_data.mat"
OUTPUT_PATH = REPO_ROOT / "PPO1_ID_FD_PhysX/models/fd/best_fd_cylinder_rank4.pt"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_fd_common() -> tuple[ModuleType, Path]:
    # Loading this small module by path avoids importing PPO1_ID_FD_PhysX/__init__.py,
    # which registers IsaacLab tasks and therefore requires a running Isaac app.
    model_refs = _load_module(
        "_fd_rank_model_refs", REPO_ROOT / "PPO1_ID_FD_PhysX/models/__init__.py"
    )
    checkpoint = Path(model_refs.CYLINDER_MODELS_ROOT) / model_refs.OLD_FD_CHECKPOINT
    common_path = (
        Path(model_refs.CYLINDER_MODELS_ROOT)
        / "fd_cylinder_space "
        / "fd_cylinder_common.py"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Original FD checkpoint not found: {checkpoint}")
    if not common_path.is_file():
        raise FileNotFoundError(f"FD common module not found: {common_path}")
    return _load_module("fd_cylinder_common", common_path), checkpoint


def _as_numpy(normalizers: dict[str, torch.Tensor], name: str) -> np.ndarray:
    return normalizers[name].detach().cpu().numpy().reshape(1, -1)


def _normalize(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return np.ascontiguousarray((x - mean) / std, dtype=np.float32)


def compute_partial_effect(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return B with shape (12, 4) using the handoff's residual regression."""
    valve = np.asarray(x[:, :4], dtype=np.float64)
    state = np.asarray(x[:, 4:], dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    state_to_target, *_ = np.linalg.lstsq(state, target, rcond=None)
    residual_target = target - state @ state_to_target
    state_to_valve, *_ = np.linalg.lstsq(state, valve, rcond=None)
    residual_valve = valve - state @ state_to_valve
    valve_to_target, *_ = np.linalg.lstsq(residual_valve, residual_target, rcond=None)
    return np.ascontiguousarray(valve_to_target.T, dtype=np.float32)


def valve_jacobian(
    model: torch.nn.Module, x: torch.Tensor, eps: float
) -> torch.Tensor:
    """Central-difference d(normalized output)/d(normalized valve)."""
    columns = []
    for valve_index in range(4):
        xp = x.clone()
        xm = x.clone()
        xp[:, valve_index] += eps
        xm[:, valve_index] -= eps
        columns.append((model(xp) - model(xm)) / (2.0 * eps))
    return torch.stack(columns, dim=2)


@torch.no_grad()
def validation_mse(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        yb = torch.from_numpy(y[start : start + batch_size]).to(device)
        squared_error += torch.sum((model(xb) - yb).square()).item()
        count += yb.numel()
    return squared_error / count


@torch.no_grad()
def mean_jacobian_metrics(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    eps: float,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    model.eval()
    jac_sum = torch.zeros((12, 4), dtype=torch.float64, device=device)
    count = min(len(x), max_samples)
    for start in range(0, count, batch_size):
        xb = torch.from_numpy(x[start : min(start + batch_size, count)]).to(device)
        jac_sum += valve_jacobian(model, xb, eps).double().sum(dim=0)
    mean_jac = (jac_sum / count).float().cpu().numpy()
    extension_singular_values = np.linalg.svd(mean_jac[:4], compute_uv=False)
    rank = int(np.sum(extension_singular_values > 0.01 * extension_singular_values[0]))
    cond = float(extension_singular_values[0] / max(extension_singular_values[-1], 1e-12))
    return mean_jac, extension_singular_values, rank, cond


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-jac", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--jac-metric-samples", type=int, default=32768)
    parser.add_argument(
        "--max-val-regression",
        type=float,
        default=0.0,
        help="Allowed fractional normalized validation-MSE regression (default: none).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.eps <= 0 or args.lambda_jac < 0:
        raise ValueError("epochs/batch-size/eps must be positive and lambda-jac nonnegative")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )

    common, checkpoint_path = _load_fd_common()
    model, checkpoint, normalizers = common.load_model_from_checkpoint(
        checkpoint_path, device=device
    )
    data = common.load_fd_data(args.data)
    if data.x_train is None or data.y_train is None or data.x_val is None or data.y_val is None:
        raise RuntimeError("Original FD data did not contain both train and validation splits")

    # The checkpoint values are authoritative; the MAT-file values are diagnostic only.
    x_mean = _as_numpy(normalizers, "x_mean")
    x_std = _as_numpy(normalizers, "x_std")
    y_mean = _as_numpy(normalizers, "y_mean")
    y_std = _as_numpy(normalizers, "y_std")
    x_train = _normalize(data.x_train, x_mean, x_std)
    y_train = _normalize(data.y_train, y_mean, y_std)
    x_val = _normalize(data.x_val, x_mean, x_std)
    y_val = _normalize(data.y_val, y_mean, y_std)

    b_numpy = compute_partial_effect(x_train, y_train)
    b_singular_values = np.linalg.svd(b_numpy[:4], compute_uv=False)
    b_rank = int(np.sum(b_singular_values > 0.01 * b_singular_values[0]))
    b_cond = float(b_singular_values[0] / max(b_singular_values[-1], 1e-12))
    print(f"device={device} checkpoint={checkpoint_path}", flush=True)
    print(f"data train={x_train.shape} val={x_val.shape}", flush=True)
    print(
        "B DeltaExtension singular values="
        f"{np.array2string(b_singular_values, precision=6)} rank={b_rank}/4 cond={b_cond:.3f}",
        flush=True,
    )
    if b_rank != 4 or b_cond > 5.0:
        raise RuntimeError(
            f"Empirical DeltaExtension partial effect failed sanity check: rank={b_rank}, cond={b_cond:.3f}"
        )

    baseline_mse = validation_mse(model, x_val, y_val, device, args.batch_size)
    _, baseline_sv, baseline_rank, baseline_cond = mean_jacobian_metrics(
        model, x_val, device, args.batch_size, args.eps, args.jac_metric_samples
    )
    print(
        f"baseline val_mse={baseline_mse:.9g} mean_J_ext_sv="
        f"{np.array2string(baseline_sv, precision=5)} rank={baseline_rank}/4 cond={baseline_cond:.2f}",
        flush=True,
    )

    b = torch.from_numpy(b_numpy).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    allowed_mse = baseline_mse * (1.0 + args.max_val_regression)
    best: dict[str, Any] | None = None

    # eval() deliberately disables the checkpoint's dropout while retaining gradients.
    # Independent dropout masks in the +eps/-eps forwards would otherwise dominate a
    # finite difference with eps=1e-2.
    model.eval()
    for epoch in range(1, args.epochs + 1):
        order = np.random.permutation(len(x_train))
        mse_total = 0.0
        jac_total = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            xb = torch.from_numpy(x_train[indices]).to(device)
            yb = torch.from_numpy(y_train[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            mse_loss = torch.mean((prediction - yb).square())
            jac = valve_jacobian(model, xb, args.eps)
            jac_loss = torch.mean((jac - b.unsqueeze(0)).square())
            loss = mse_loss + args.lambda_jac * jac_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            batch_count = len(indices)
            mse_total += mse_loss.item() * batch_count
            jac_total += jac_loss.item() * batch_count
            seen += batch_count

        val_mse = validation_mse(model, x_val, y_val, device, args.batch_size)
        mean_jac, singular_values, rank, cond = mean_jacobian_metrics(
            model, x_val, device, args.batch_size, args.eps, args.jac_metric_samples
        )
        print(
            f"epoch={epoch:02d} train_mse={mse_total / seen:.9g} "
            f"train_jac={jac_total / seen:.9g} val_mse={val_mse:.9g} "
            f"J_ext_sv={np.array2string(singular_values, precision=5)} "
            f"rank={rank}/4 cond={cond:.2f}",
            flush=True,
        )
        eligible = val_mse <= allowed_mse and rank == 4
        score = (cond, val_mse)
        if eligible and (best is None or score < best["score"]):
            best = {
                "score": score,
                "epoch": epoch,
                "val_mse": val_mse,
                "rank": rank,
                "cond": cond,
                "singular_values": singular_values.copy(),
                "mean_jacobian": mean_jac.copy(),
                "state_dict": _cpu_state_dict(model),
            }

    if best is None:
        raise RuntimeError(
            "No epoch was full-rank without validation-MSE regression; checkpoint was not written"
        )

    saved_checkpoint = dict(checkpoint)
    saved_checkpoint["model_state_dict"] = best.pop("state_dict")
    saved_checkpoint["rank_finetune"] = {
        "source_checkpoint": str(checkpoint_path),
        "data_path": str(args.data.resolve()),
        "lambda_jac": args.lambda_jac,
        "finite_difference_eps": args.eps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "baseline_val_mse_normalized": baseline_mse,
        "partial_effect_B_normalized": b_numpy,
        **best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved_checkpoint, args.output)
    print(
        f"saved={args.output} epoch={best['epoch']} val_mse={best['val_mse']:.9g} "
        f"rank={best['rank']}/4 cond={best['cond']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
