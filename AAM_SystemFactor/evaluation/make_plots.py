"""Generate evaluation plots for AAM SystemFactor."""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def make_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_nominal: np.ndarray,
    run_types: List[str],
    history: Dict,
    output_dir: str,
    target_names: List[str],
) -> None:
    """Generate all required evaluation plots."""
    if not HAS_MPL:
        print("Matplotlib not available, skipping plots")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_factors = y_true.shape[1]

    # 1. Ground truth vs prediction
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for k in range(n_factors):
        ax = axes[k]
        ax.scatter(y_true[:, k], y_pred[:, k], alpha=0.3, s=2)
        lims = [min(y_true[:, k].min(), y_pred[:, k].min()),
                max(y_true[:, k].max(), y_pred[:, k].max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        ax.set_title(target_names[k])
    axes[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '01_ground_truth_vs_prediction.png'), dpi=100)
    plt.close(fig)

    # 2. Residual distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    residuals = y_pred - y_true
    for k in range(n_factors):
        ax = axes[k]
        ax.hist(residuals[:, k], bins=50, alpha=0.7)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')
        ax.set_title(target_names[k])
    axes[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '02_residual_distributions.png'), dpi=100)
    plt.close(fig)

    # 8. Training curves
    if history and 'epoch' in history and len(history['epoch']) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history['epoch'], history['train_loss'], label='Train')
        axes[0].plot(history['epoch'], history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Training and Validation Loss')

        axes[1].plot(history['epoch'], history['lr'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('LR')
        axes[1].set_title('Learning Rate')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '10_training_curves.png'), dpi=100)
        plt.close(fig)

    print(f"Plots saved to {output_dir}")
