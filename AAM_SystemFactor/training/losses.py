"""Loss functions for AAM factor estimation."""

import torch
import torch.nn as nn


class FactorLoss(nn.Module):
    """Weighted factor MSE loss: sum_k w_k (e_hat_k - e_k)^2"""

    def __init__(self, weights=None):
        super().__init__()
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = torch.ones(7)

    def to(self, device):
        self.weights = self.weights.to(device)
        return super().to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.weights = self.weights.to(pred.device)
        sq_err = (pred - target) ** 2
        return (sq_err * self.weights).mean()


class SmoothnessLoss(nn.Module):
    """Temporal smoothness for consecutive windows from the same run."""

    def forward(self, pred: torch.Tensor, pred_prev: torch.Tensor) -> torch.Tensor:
        return ((pred - pred_prev) ** 2).mean()


class CombinedLoss(nn.Module):
    """L_total = L_factor + lambda_smooth * L_smooth"""

    def __init__(self, lambda_smooth: float = 0.01, factor_weights=None):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.factor_loss = FactorLoss(factor_weights)
        self.smooth_loss = SmoothnessLoss()

    def forward(self, pred, target, pred_prev=None):
        loss_factor = self.factor_loss(pred, target)
        loss_total = loss_factor
        loss_smooth_val = 0.0

        if pred_prev is not None and self.lambda_smooth > 0:
            loss_smooth = self.smooth_loss(pred, pred_prev)
            loss_total = loss_factor + self.lambda_smooth * loss_smooth
            loss_smooth_val = loss_smooth.item()

        return {'loss': loss_total, 'factor': loss_factor.item(), 'smooth': loss_smooth_val}
