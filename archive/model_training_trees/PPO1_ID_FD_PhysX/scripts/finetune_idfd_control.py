#!/usr/bin/env python3
"""Jointly fine-tune ID/FD for target intervention while preserving data MSE."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def norm(raw: torch.Tensor, stats: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    mean = stats[f"{prefix}_mean"].to(raw.device).reshape(1, -1)
    std = stats[f"{prefix}_std"].to(raw.device).reshape(1, -1).clamp_min(1e-8)
    return (raw - mean) / std


def denorm(value: torch.Tensor, stats: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    mean = stats[f"{prefix}_mean"].to(value.device).reshape(1, -1)
    std = stats[f"{prefix}_std"].to(value.device).reshape(1, -1).clamp_min(1e-8)
    return value * std + mean


@torch.no_grad()
def metrics(id_model, fd_model, arrays, stats, device, count=32768):
    id_x, id_y, fd_x, fd_y = arrays
    n = min(count, len(id_x))
    idx = np.arange(n)
    perm = np.random.default_rng(123).permutation(n)
    batch = 4096
    id_se = fd_se = control_se = 0.0
    pred_all, goal_all = [], []
    for start in range(0, n, batch):
        ii = idx[start:start + batch]
        pp = perm[start:start + batch]
        ix = torch.from_numpy(id_x[ii]).to(device)
        iy = torch.from_numpy(id_y[ii]).to(device)
        fx = torch.from_numpy(fd_x[ii]).to(device)
        fy = torch.from_numpy(fd_y[ii]).to(device)
        id_pred_n = id_model(norm(ix, stats["id"], "x"))
        fd_pred_n = fd_model(norm(fx, stats["fd"], "x"))
        id_se += torch.sum((id_pred_n - norm(iy, stats["id"], "y")) ** 2).item()
        fd_se += torch.sum((fd_pred_n - norm(fy, stats["fd"], "y")) ** 2).item()

        synth = ix.clone()
        synth[:, 16:24] = torch.from_numpy(id_x[pp, 16:24]).to(device)
        zero = synth.clone(); zero[:, 16:24] = 0.0
        prev = synth[:, 24:28]
        valve = denorm(id_model(norm(synth, stats["id"], "x")), stats["id"], "y") + prev
        valve0 = denorm(id_model(norm(zero, stats["id"], "x")), stats["id"], "y") + prev
        state = synth[:, :16]
        out = denorm(fd_model(norm(torch.cat([valve, state], 1), stats["fd"], "x")), stats["fd"], "y")
        out0 = denorm(fd_model(norm(torch.cat([valve0, state], 1), stats["fd"], "x")), stats["fd"], "y")
        pred = out[:, :4] - out0[:, :4]
        goal = synth[:, 16:20]
        scale = stats["fd"]["y_std"].to(device).reshape(1, -1)[:, :4].clamp_min(1e-8)
        control_se += torch.sum(((pred - goal) / scale) ** 2).item()
        pred_all.append(pred.cpu()); goal_all.append(goal.cpu())
    pred = torch.cat(pred_all); goal = torch.cat(goal_all)
    slopes, corrs = [], []
    for axis in range(4):
        slopes.append(float(torch.dot(pred[:, axis], goal[:, axis]) / torch.dot(goal[:, axis], goal[:, axis])))
        corrs.append(float(torch.corrcoef(torch.stack([pred[:, axis], goal[:, axis]]))[0, 1]))
    return {
        "id_mse": id_se / (n * 4), "fd_mse": fd_se / (n * 12),
        "control_mse": control_se / (n * 4), "slopes": slopes, "corrs": corrs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lambda-control", type=float, default=1.0)
    p.add_argument("--max-val-regression", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)

    refs = load_module("_control_model_refs", ROOT / "PPO1_ID_FD_PhysX/models/__init__.py")
    model_root = Path(refs.CYLINDER_MODELS_ROOT)
    id_dir = model_root / "id_cylinder_space "
    fd_dir = model_root / "fd_cylinder_space "
    id_common = load_module("id_cylinder_common", id_dir / "id_cylinder_common.py")
    fd_common = load_module("fd_cylinder_common", fd_dir / "fd_cylinder_common.py")
    id_path = model_root / refs.OLD_ID_CHECKPOINT
    fd_path = ROOT / "PPO1_ID_FD_PhysX/models/fd/best_fd_cylinder_rank4.pt"
    id_model, id_ckpt, id_stats = id_common.load_model_from_checkpoint(id_path, device=device)
    fd_model, fd_ckpt, fd_stats = fd_common.load_model_from_checkpoint(fd_path, device=device)
    id_data = id_common.load_id_data(ROOT / "PPO1_ID_FD_PhysX/models/id/ID_cylinder_data.mat")
    fd_data = fd_common.load_fd_data(ROOT / "PPO1_ID_FD_PhysX/models/fd/FD_cylinder_data.mat")
    id_train_x = id_common.prepare_features_for_checkpoint(id_data, id_ckpt, split="train")
    id_val_x = id_common.prepare_features_for_checkpoint(id_data, id_ckpt, split="val")
    id_train_y = id_data.y_train - id_train_x[:, 24:28]
    id_val_y = id_data.y_val - id_val_x[:, 24:28]
    train = tuple(np.ascontiguousarray(x, dtype=np.float32) for x in (
        id_train_x, id_train_y, fd_data.x_train, fd_data.y_train))
    val = tuple(np.ascontiguousarray(x, dtype=np.float32) for x in (
        id_val_x, id_val_y, fd_data.x_val, fd_data.y_val))
    stats = {"id": id_stats, "fd": fd_stats}
    baseline = metrics(id_model, fd_model, val, stats, device)
    print(f"baseline={baseline}", flush=True)

    id_model.eval(); fd_model.eval()
    optimizer = torch.optim.Adam(list(id_model.parameters()) + list(fd_model.parameters()), lr=a.lr)
    best = None
    for epoch in range(1, a.epochs + 1):
        order = np.random.permutation(len(train[0]))
        totals = np.zeros(3); seen = 0
        for start in range(0, len(order), a.batch_size):
            ii = order[start:start + a.batch_size]
            pp = np.random.randint(0, len(train[0]), size=len(ii))
            ix = torch.from_numpy(train[0][ii]).to(device)
            iy = torch.from_numpy(train[1][ii]).to(device)
            fx = torch.from_numpy(train[2][ii]).to(device)
            fy = torch.from_numpy(train[3][ii]).to(device)
            optimizer.zero_grad(set_to_none=True)
            id_loss = torch.mean((id_model(norm(ix, id_stats, "x")) - norm(iy, id_stats, "y")) ** 2)
            fd_loss = torch.mean((fd_model(norm(fx, fd_stats, "x")) - norm(fy, fd_stats, "y")) ** 2)
            synth = ix.clone()
            synth[:, 16:24] = torch.from_numpy(train[0][pp, 16:24]).to(device)
            zero = synth.clone(); zero[:, 16:24] = 0.0
            prev = synth[:, 24:28]
            valve = denorm(id_model(norm(synth, id_stats, "x")), id_stats, "y") + prev
            valve0 = denorm(id_model(norm(zero, id_stats, "x")), id_stats, "y") + prev
            state = synth[:, :16]
            out = denorm(fd_model(norm(torch.cat([valve, state], 1), fd_stats, "x")), fd_stats, "y")
            out0 = denorm(fd_model(norm(torch.cat([valve0, state], 1), fd_stats, "x")), fd_stats, "y")
            scale = fd_stats["y_std"].to(device).reshape(1, -1)[:, :4].clamp_min(1e-8)
            control_loss = torch.mean(((out[:, :4] - out0[:, :4] - synth[:, 16:20]) / scale) ** 2)
            loss = id_loss + fd_loss + a.lambda_control * control_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(id_model.parameters()) + list(fd_model.parameters()), 5.0)
            optimizer.step()
            totals += [id_loss.item() * len(ii), fd_loss.item() * len(ii), control_loss.item() * len(ii)]
            seen += len(ii)
        result = metrics(id_model, fd_model, val, stats, device)
        print(f"epoch={epoch} train={(totals/seen).tolist()} val={result}", flush=True)
        eligible = (
            result["id_mse"] <= baseline["id_mse"] * (1 + a.max_val_regression)
            and result["fd_mse"] <= baseline["fd_mse"] * (1 + a.max_val_regression)
        )
        score = result["control_mse"]
        if eligible and (best is None or score < best[0]):
            best = (score, result, {k: v.detach().cpu() for k, v in id_model.state_dict().items()},
                    {k: v.detach().cpu() for k, v in fd_model.state_dict().items()}, epoch)
    if best is None:
        raise RuntimeError("No control checkpoint preserved both validation MSE gates")
    _, result, id_state, fd_state, epoch = best
    id_out = dict(id_ckpt); id_out["model_state_dict"] = id_state
    fd_out = dict(fd_ckpt); fd_out["model_state_dict"] = fd_state
    metadata = {"epoch": epoch, "baseline": baseline, "result": result, "objective": "shuffled_target_paired_residual"}
    id_out["control_finetune"] = metadata; fd_out["control_finetune"] = metadata
    id_path_out = ROOT / "PPO1_ID_FD_PhysX/models/id/best_id_cylinder_control.pt"
    fd_path_out = ROOT / "PPO1_ID_FD_PhysX/models/fd/best_fd_cylinder_control.pt"
    torch.save(id_out, id_path_out); torch.save(fd_out, fd_path_out)
    print(f"saved id={id_path_out} fd={fd_path_out} result={result}", flush=True)


if __name__ == "__main__":
    main()
