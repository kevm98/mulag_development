#!/usr/bin/env python3
"""Fine-tune ID model on valve-space PID teacher data."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- Model definition (module-level) ----
class VelocityToValveMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 4),
        )
    def forward(self, x):
        return self.net(x)

# ---- Training code (only runs when executed directly) ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_dynamic_dataset.npz")
    parser.add_argument("--output_dir", type=str, default="outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = np.load(args.dataset)
    X_all = data["X"].astype(np.float32); Y_all = data["Y"].astype(np.float32)
    print(f"Loaded ID dataset: X={X_all.shape}, Y={Y_all.shape}")
    np.random.seed(42); n = X_all.shape[0]
    idx = np.random.permutation(n); n_val = int(n * args.val_split)
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_val, Y_val = X_all[val_idx], Y_all[val_idx]
    X_mean = X_train.mean(axis=0, keepdims=True); X_std = X_train.std(axis=0, keepdims=True) + 1e-8
    Y_mean = Y_train.mean(axis=0, keepdims=True); Y_std = Y_train.std(axis=0, keepdims=True) + 1e-8

    model = VelocityToValveMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_mse = float("inf")

    for epoch in range(args.epochs):
        model.train(); train_losses = []
        for b in range(0, n - n_val, args.batch_size):
            bx = torch.from_numpy(X_train[b:b+args.batch_size]).to(device)
            by = torch.from_numpy(Y_train[b:b+args.batch_size]).to(device)
            bx_n = (bx - torch.from_numpy(X_mean).to(device)) / torch.from_numpy(X_std).to(device)
            yp = model(bx_n); loss = ((yp - by)**2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_losses.append(loss.item())
        model.eval(); val_preds, val_targs = [], []
        with torch.no_grad():
            for b in range(0, n_val, args.batch_size):
                bx = torch.from_numpy(X_val[b:b+args.batch_size]).to(device)
                by = torch.from_numpy(Y_val[b:b+args.batch_size]).to(device)
                yp = model((bx - torch.from_numpy(X_mean).to(device)) / torch.from_numpy(X_std).to(device))
                val_preds.append(yp.cpu().numpy()); val_targs.append(by.cpu().numpy())
        vp = np.concatenate(val_preds); vt = np.concatenate(val_targs)
        vmse = ((vp-vt)**2).mean(); vmae = np.abs(vp-vt).mean(); vsa = np.mean(np.sign(vp)==np.sign(vt))
        if vmse < best_val_mse:
            best_val_mse = vmse
            torch.save({"model_state_dict": model.state_dict(), "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std}, outdir / "id_physx_finetuned_best.pt")
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train={np.mean(train_losses):.6f} val_mse={vmse:.6f} val_mae={vmae:.4f} sign={vsa:.3f}", flush=True)
    model.eval(); all_preds = []
    with torch.no_grad():
        for b in range(0, n, args.batch_size):
            bx = torch.from_numpy(X_all[b:b+args.batch_size]).to(device)
            yp = model((bx - torch.from_numpy(X_mean).to(device)) / torch.from_numpy(X_std).to(device))
            all_preds.append(yp.cpu().numpy())
    ap = np.concatenate(all_preds)
    per_cyl_sa = [np.mean(np.sign(ap[:,i])==np.sign(Y_all[:,i])) for i in range(4)]
    per_cyl_rmse = [np.sqrt(np.mean((ap[:,i]-Y_all[:,i])**2)) for i in range(4)]
    report = [f"# ID PhysX Fine-Tuning Report", f"- best_val_mse: {best_val_mse:.6f}", f"- sign_accuracy: {per_cyl_sa}", f"- RMSE: {per_cyl_rmse}"]
    (outdir / "id_physx_finetune_report.md").write_text("\n".join(report))
    print(f"Saved: {outdir}/id_physx_finetuned_best.pt")

if __name__ == "__main__":
    main()
