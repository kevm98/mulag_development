#!/usr/bin/env python3
"""Fine-tune the FD (HistoryGRU) model on PhysX dynamic valve-response data."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/fd_physx_dynamic_dataset.npz")
parser.add_argument("--output_dir", type=str, default="outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--val_split", type=float, default=0.2)
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)

# Load dataset
data = np.load(args.dataset)
X = data["X"].astype(np.float32)
Y = data["Y"].astype(np.float32)
print(f"Loaded dataset: X={X.shape}, Y={Y.shape}")

# Train/val split
np.random.seed(42)
n = X.shape[0]
idx = np.random.permutation(n)
n_val = int(n * args.val_split)
train_idx, val_idx = idx[n_val:], idx[:n_val]
X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

# Compute normalization
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-8
Y_mean = Y_train.mean(axis=0, keepdims=True)
Y_std = Y_train.std(axis=0, keepdims=True) + 1e-8

# Load pre-trained FD checkpoint
models_dir = Path(__file__).resolve().parent / "Models_cylinder " / "fd_cylinder_space " / "runs_fd_cylinder" / "fd_cylinder_sequence_motion_v1"
ckpt_path = models_dir / "best_fd_cylinder_sequence_motion_v1.pt"
print(f"Loading FD checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Build model
from sequence_cylinder_common import HistoryGRU
model_cfg = ckpt["model_config"]
model = HistoryGRU(
    step_dim=model_cfg["step_dim"],
    current_dim=model_cfg["current_dim"],
    output_dim=model_cfg["output_dim"],
    hidden_dim=model_cfg["hidden_dim"],
    num_layers=model_cfg.get("num_layers", 1),
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.train()

# Normalize function (X: [valve(4), ext(4), extvel(4)] -> seq [B,1,12])
# FD model expects sequence of [B, H, step_dim]
def normalize_x(x_batch, history_len=10):
    x_np = (x_batch - X_mean) / X_std
    x_np = np.repeat(x_np[:, np.newaxis, :], history_len, axis=1)
    return x_np
def denormalize_y(y_batch):
    return y_batch * Y_std + Y_mean

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
best_val_loss = float("inf")

report_lines = [f"# FD PhysX Fine-Tuning Report", f"- dataset: {args.dataset}", f"- samples: {n}", f"- train/val: {n-n_val}/{n_val}"]

for epoch in range(args.epochs):
    model.train()
    train_losses = []
    for b in range(0, n - n_val, args.batch_size):
        batch_x = torch.from_numpy(X_train[b:b+args.batch_size]).to(device)
        batch_y = torch.from_numpy(Y_train[b:b+args.batch_size]).to(device)
        seq_x = normalize_x(batch_x.cpu().numpy())
        seq_x_t = torch.from_numpy(seq_x).to(device)
        y_pred = model(seq_x_t, None)
        loss = ((y_pred - batch_y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for b in range(0, n_val, args.batch_size):
            bx = torch.from_numpy(X_val[b:b+args.batch_size]).to(device)
            by = torch.from_numpy(Y_val[b:b+args.batch_size]).to(device)
            sx = normalize_x(bx.cpu().numpy())
            sxt = torch.from_numpy(sx).to(device)
            yp = model(sxt, None)
            val_preds.append(yp.cpu().numpy())
            val_targets.append(by.cpu().numpy())
    vp = np.concatenate(val_preds, axis=0)
    vt = np.concatenate(val_targets, axis=0)
    val_mse = ((vp - vt) ** 2).mean()
    val_mae = np.abs(vp - vt).mean()
    
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": model_cfg,
            "X_mean": X_mean, "X_std": X_std,
            "Y_mean": Y_mean, "Y_std": Y_std,
        }, outdir / "fd_physx_finetuned_best.pt")
    
    if epoch % 10 == 0:
        # Sign accuracy
        sign_acc = np.mean(np.sign(vp) == np.sign(vt))
        print(f"Epoch {epoch}: train_loss={np.mean(train_losses):.6f} val_mse={val_mse:.6f} val_mae={val_mae:.6f} sign_acc={sign_acc:.3f}", flush=True)

# Final metrics
model.eval()
all_preds = []
with torch.no_grad():
    for b in range(0, n, args.batch_size):
        bx = torch.from_numpy(X[b:b+args.batch_size]).to(device)
        sx = normalize_x(bx.cpu().numpy())
        sxt = torch.from_numpy(sx).to(device)
        all_preds.append(model(sxt, None).cpu().numpy())
ap = np.concatenate(all_preds, axis=0)
per_cyl_sign = [np.mean(np.sign(ap[:, i]) == np.sign(Y[:, i])) for i in range(4)]
per_cyl_mae = [np.abs(ap[:, i] - Y[:, i]).mean() for i in range(4)]
per_cyl_rmse = [np.sqrt(np.mean((ap[:, i] - Y[:, i])**2)) for i in range(4)]

report_lines += [
    f"- epochs: {args.epochs}, lr: {args.lr}",
    f"- best_val_mse: {best_val_loss:.6f}",
    f"- final_mse: {((ap - Y)**2).mean():.6f}",
    f"- sign_accuracy per cylinder: {per_cyl_sign}",
    f"- MAE per cylinder: {per_cyl_mae}",
    f"- RMSE per cylinder: {per_cyl_rmse}",
    f"- delta_pred mean abs: {np.abs(ap).mean(axis=0)}",
    f"- delta_true mean abs: {np.abs(Y).mean(axis=0)}",
]

# Save scaler
np.savez(outdir / "fd_physx_scaler.npz", X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)

(outdir / "fd_physx_finetune_report.md").write_text("\n".join(report_lines))
print(f"Fine-tuning complete. Saved to {outdir}/fd_physx_finetuned_best.pt")
