"""Evaluation metrics for AAM SystemFactor predictions."""

import numpy as np
from scipy import stats
from typing import Dict, List


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute per-factor and aggregate evaluation metrics.

    Args:
        y_true: (N, 7) ground truth targets
        y_pred: (N, 7) predictions

    Returns:
        dict with per-factor arrays and aggregate scalars
    """
    n_factors = y_true.shape[1]
    mae = np.zeros(n_factors)
    rmse = np.zeros(n_factors)
    nrmse = np.zeros(n_factors)
    r2 = np.zeros(n_factors)
    pearson = np.zeros(n_factors)
    spearman = np.zeros(n_factors)
    bias = np.zeros(n_factors)
    error_std = np.zeros(n_factors)

    for k in range(n_factors):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        err = yp - yt

        mae[k] = np.mean(np.abs(err))
        rmse[k] = np.sqrt(np.mean(err ** 2))
        yt_std = np.std(yt)
        nrmse[k] = rmse[k] / yt_std if yt_std > 1e-8 else np.nan
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2[k] = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else np.nan

        if len(yt) >= 3:
            pearson[k], _ = stats.pearsonr(yt, yp)
            spearman[k], _ = stats.spearmanr(yt, yp)
        else:
            pearson[k] = np.nan
            spearman[k] = np.nan

        bias[k] = np.mean(err)
        error_std[k] = np.std(err)

    return {
        'mae': mae,
        'rmse': rmse,
        'nrmse': nrmse,
        'r2': r2,
        'pearson': pearson,
        'spearman': spearman,
        'bias': bias,
        'error_std': error_std,
    }


def evaluate_by_subgroup(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_types: np.ndarray,
) -> Dict[str, Dict]:
    """Compute metrics stratified by run type."""
    subgroups = {}
    for rt in np.unique(run_types):
        mask = run_types == rt
        if mask.sum() >= 3:
            subgroups[rt] = compute_metrics(y_true[mask], y_pred[mask])
    subgroups['all'] = compute_metrics(y_true, y_pred)
    return subgroups
