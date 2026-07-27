"""Main training script for AAM SystemFactor LSTM."""

import os
import sys
import json
import csv
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AAM_SystemFactor.data.matlab_v73 import load_open_loop_run, load_closed_loop_run, compute_ext_vel_causal
from AAM_SystemFactor.data.factor_targets import compute_targets, NOMINAL_TARGET, TARGET_NAMES
from AAM_SystemFactor.data.cylinder_windows import build_feature_matrix, build_windows_for_runs
from AAM_SystemFactor.data.normalization import FeatureScaler, TargetScaler
from AAM_SystemFactor.data.split_runs import split_runs_deterministic, save_split_manifest, verify_no_leakage
from AAM_SystemFactor.models.aam_lstm import AAMLSTM
from AAM_SystemFactor.models.baselines import NominalBaseline, AggregateMLP
from AAM_SystemFactor.training.trainer import AAMTrainer
from AAM_SystemFactor.evaluation.evaluate_aam import compute_metrics, evaluate_by_subgroup
from AAM_SystemFactor.evaluation.make_plots import make_all_plots

# Paths
THESIS_BASE = os.path.expanduser(
    "~/rrlab_mulag/thesis /thesis/hydraulic_manipulator_thesis/"
    "Domain_Randomization_Data_Collection-20260720T142849Z-1-001/"
    "Domain_Randomization_Data_Collection"
)
OL_RUNS_DIR = os.path.join(THESIS_BASE, "Randomization_open_loop/Excitation_Finetune_25/runs")
CL_MAT = os.path.join(THESIS_BASE, "Randomization_closed_loop/Hydraulic_Mechanic_Dataset_Curriculum_1000.mat")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')


def load_all_runs(T: int = 10, max_cl_runs: int = None):
    """Load all open-loop and closed-loop runs."""
    runs_data = []
    run_ids = []
    source_types = []
    run_types = []
    params_list = []
    n_samples_list = []

    # Open-loop
    ol_files = sorted([f for f in os.listdir(OL_RUNS_DIR) if f.endswith('.mat')])
    for fname in ol_files:
        path = os.path.join(OL_RUNS_DIR, fname)
        rd = load_open_loop_run(path)
        if rd['ext_vel'] is None:
            rd['ext_vel'] = compute_ext_vel_causal(rd['extension'], rd['dt'])
        runs_data.append(rd)
        run_ids.append(fname.replace('.mat', ''))
        source_types.append('open_loop')
        run_types.append(rd['run_type'])
        params_list.append(rd['params'])
        n_samples_list.append(rd['n_transitions'])

    # Closed-loop
    n_cl = max_cl_runs if max_cl_runs else 1000
    for i in range(n_cl):
        rd = load_closed_loop_run(CL_MAT, i)
        if rd['ext_vel'] is None:
            rd['ext_vel'] = compute_ext_vel_causal(rd['extension'], rd['dt'])
        runs_data.append(rd)
        run_ids.append(f'cl_{i:04d}')
        source_types.append('closed_loop')
        run_types.append(rd['run_type'])
        params_list.append(rd['params'])
        n_samples_list.append(rd['n_transitions'])

    # Compute targets
    targets = np.array([compute_targets(p) for p in params_list])

    # Compute windows
    n_windows_list = []
    for rd in runs_data:
        features = build_feature_matrix(rd)
        n_w = max(0, rd['n_transitions'] - T + 1)
        n_windows_list.append(n_w)

    return runs_data, targets, run_ids, source_types, run_types, n_samples_list, n_windows_list


def main(T: int = 10, epochs: int = 100, batch_size: int = 256, warmup: bool = False):
    """Run training pipeline.

    Args:
        T: history length
        epochs: number of training epochs
        batch_size: training batch size
        warmup: if True, run only 5 epochs as smoke test
    """
    print(f"=== AAM SystemFactor Training ===\nT={T}, epochs={epochs}, batch_size={batch_size}")

    # 1. Load data
    print("\n[1/7] Loading datasets...")
    t0 = time.time()
    max_cl = 100 if warmup else None
    runs_data, targets, run_ids, source_types, run_types, n_samples, n_windows = load_all_runs(T=T, max_cl_runs=max_cl)
    print(f"  Loaded {len(runs_data)} runs in {time.time()-t0:.1f}s")

    # 2. Split runs
    print("\n[2/7] Splitting runs...")
    train_mask, val_mask, test_mask, assignments = split_runs_deterministic(
        run_ids, run_types, source_types, n_samples, train_frac=0.80, val_frac=0.10, seed=42
    )
    train_runs = np.where(train_mask)[0]
    val_runs = np.where(val_mask)[0]
    test_runs = np.where(test_mask)[0]
    print(f"  Train: {len(train_runs)}, Val: {len(val_runs)}, Test: {len(test_runs)}")

    save_split_manifest(
        os.path.join(OUTPUT_DIR, 'run_split_manifest.csv'),
        run_ids, source_types, run_types, assignments, n_samples, n_windows,
    )

    # Verify no leakage
    verify_no_leakage(set(train_runs), set(val_runs), set(test_runs))
    print("  Zero split leakage verified")

    # 3. Build windows
    print("\n[3/7] Building windows...")
    t0 = time.time()

    def build_split_windows(run_indices):
        w_list, rid_list = [], []
        for ri in run_indices:
            features = build_feature_matrix(runs_data[ri])
            n = features.shape[0]
            if n >= T:
                for j in range(n - T + 1):
                    w_list.append(features[j:j + T])
                    rid_list.append(ri)
        if not w_list:
            return np.empty((0, T, 20)), np.array([], dtype=int)
        return np.stack(w_list), np.array(rid_list)

    X_train, rid_train = build_split_windows(train_runs)
    X_val, rid_val = build_split_windows(val_runs)
    X_test, rid_test = build_split_windows(test_runs)

    y_train = targets[rid_train]
    y_val = targets[rid_val]
    y_test = targets[rid_test]

    print(f"  Train windows: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Build time: {time.time()-t0:.1f}s")

    # 4. Normalize
    print("\n[4/7] Fitting scalers (train only)...")
    # Reshape windows to (N*T, 20) for per-feature fitting
    X_train_flat = X_train.reshape(-1, 20)
    input_scaler = FeatureScaler()
    input_scaler.fit(X_train_flat)

    # Target standardization
    target_scaler = TargetScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    # Scale input windows
    X_train_scaled = np.zeros_like(X_train)
    X_val_scaled = np.zeros_like(X_val)
    X_test_scaled = np.zeros_like(X_test)
    for i in range(len(X_train)):
        X_train_scaled[i] = input_scaler.transform(X_train[i])
    for i in range(len(X_val)):
        X_val_scaled[i] = input_scaler.transform(X_val[i])
    for i in range(len(X_test)):
        X_test_scaled[i] = input_scaler.transform(X_test[i])

    input_scaler.save(os.path.join(OUTPUT_DIR, 'input_scaler.json'))
    target_scaler.save(os.path.join(OUTPUT_DIR, 'target_scaler.json'))
    print(f"  Input mean range: [{input_scaler.mean.min():.2f}, {input_scaler.mean.max():.2f}]")
    print(f"  Input std range:  [{input_scaler.std.min():.2f}, {input_scaler.std.max():.2f}]")

    # 5. Build model
    print("\n[5/7] Building model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    model = AAMLSTM(input_size=20, hidden_size=64, num_layers=2, dropout=0.2, output_size=7)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Train
    print("\n[6/7] Training...")
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val_scaled, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    actual_epochs = 5 if warmup else epochs
    trainer = AAMTrainer(
        model, device, lr=1e-3, weight_decay=1e-4, grad_clip=1.0,
        patience=min(15, actual_epochs // 2), checkpoint_dir=os.path.join(OUTPUT_DIR, '..'),
    )
    trainer.fit(train_loader, val_loader, epochs=actual_epochs)

    # Save training history
    history_path = os.path.join(OUTPUT_DIR, 'training_history.csv')
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_factor', 'val_factor', 'lr', 'epoch_time_s'])
        for i in range(len(trainer.history['epoch'])):
            writer.writerow([
                trainer.history['epoch'][i],
                trainer.history['train_loss'][i],
                trainer.history['val_loss'][i],
                trainer.history['train_factor'][i],
                trainer.history['val_factor'][i],
                trainer.history['lr'][i],
                trainer.history['epoch_time_s'][i],
            ])

    # 7. Evaluate
    print("\n[7/7] Evaluating...")
    model.eval()
    with torch.no_grad():
        x_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        y_pred_scaled = model(x_test_t).cpu().numpy()

    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    # Nominal baseline
    nominal_pred = np.tile(NOMINAL_TARGET, (len(y_test), 1))
    nominal_metrics = compute_metrics(y_test, nominal_pred)
    model_metrics = compute_metrics(y_test, y_pred)

    print("\nNominal baseline RMSE:", nominal_metrics['rmse'])
    print("LSTM model RMSE:      ", model_metrics['rmse'])
    print(f"\nImprovement: {(1 - model_metrics['rmse'].mean() / nominal_metrics['rmse'].mean()) * 100:.1f}%")

    # Determine if useful
    lstm_mean_rmse = np.mean(model_metrics['rmse'])
    nominal_mean_rmse = np.mean(nominal_metrics['rmse'])
    useful = lstm_mean_rmse < nominal_mean_rmse

    # Make plots
    make_all_plots(
        y_test, y_pred, nominal_pred, run_types,
        trainer.history, os.path.join(OUTPUT_DIR, 'plots'),
        TARGET_NAMES,
    )

    result = {
        'lstm_rmse': model_metrics['rmse'].tolist(),
        'nominal_rmse': nominal_metrics['rmse'].tolist(),
        'useful': bool(useful),
        'warmup': warmup,
        'T': T,
    }
    with open(os.path.join(OUTPUT_DIR, 'evaluation_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete. Useful: {useful}")
    print(f"Results saved to {OUTPUT_DIR}")
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--warmup', action='store_true')
    args = parser.parse_args()
    main(T=args.T, epochs=args.epochs, batch_size=args.batch_size, warmup=args.warmup)
