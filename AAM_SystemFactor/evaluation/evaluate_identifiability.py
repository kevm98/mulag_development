"""Identifiability analysis: evaluate which factors are learnable from observable signals."""

import numpy as np
from scipy import stats
from typing import Dict, List


def identifiability_audit(
    features: np.ndarray,
    run_types: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
) -> Dict:
    """Evaluate per-factor identifiability from causal observable statistics.

    Args:
        features: (N, T, 20) input windows
        run_types: (N,) run type strings
        targets: (N, 7) ground truth factors
        target_names: list of factor names

    Returns:
        dict with per-factor statistics
    """
    N, T, F = features.shape
    n_factors = len(target_names)

    # Compute summary statistics per window
    stats_feat = np.zeros((N, F * 5))  # mean, std, min, max, last
    for i in range(N):
        win = features[i]
        s = []
        for f in range(F):
            ch = win[:, f]
            s.extend([np.mean(ch), np.std(ch), np.min(ch), np.max(ch), ch[-1]])
        stats_feat[i] = s

    report = {}
    for k in range(n_factors):
        y = targets[:, k]
        correlations = []
        for j in range(stats_feat.shape[1]):
            if np.std(stats_feat[:, j]) > 1e-10 and np.std(y) > 1e-10:
                r, _ = stats.pearsonr(stats_feat[:, j], y)
                correlations.append((j, abs(r)))

        correlations.sort(key=lambda x: -x[1])
        top5 = correlations[:5]

        y_var = np.var(y)

        # Classification
        if y_var < 1e-8:
            cls = 'unidentifiable (zero variance)'
        elif top5 and top5[0][1] > 0.3:
            cls = 'learnable'
        elif top5 and top5[0][1] > 0.1:
            cls = 'weakly learnable'
        else:
            cls = 'unidentifiable'

        report[target_names[k]] = {
            'variance': float(y_var),
            'top_correlations': [(int(j), float(r)) for j, r in top5],
            'classification': cls,
        }

    return report
