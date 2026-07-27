"""Training loop for AAM SystemFactor models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import time

from AAM_SystemFactor.training.losses import CombinedLoss


class AAMTrainer:
    """Training loop with early stopping, checkpointing, and history logging."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        patience: int = 15,
        checkpoint_dir: str = 'checkpoints',
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = CombinedLoss(lambda_smooth=0.0)

        self.history: Dict[str, list] = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_factor': [], 'val_factor': [],
            'lr': [], 'epoch_time_s': [],
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """Run full training loop."""
        best_val_loss = float('inf')
        best_epoch = 0
        no_improve = 0

        for epoch in range(epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            elapsed = time.time() - t0

            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['loss'])

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_factor'].append(train_metrics['factor'])
            self.history['val_factor'].append(val_metrics['factor'])
            self.history['lr'].append(current_lr)
            self.history['epoch_time_s'].append(elapsed)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train: {train_metrics['loss']:.4f} | "
                    f"Val: {val_metrics['loss']:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                no_improve = 0
                self._save_checkpoint('aam_system_factor_best.pt', epoch, val_metrics)
            else:
                no_improve += 1

            if no_improve >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        self._save_checkpoint('aam_system_factor_last.pt', epoch, val_metrics)

        return {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
        }

    def _train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        total_factor = 0.0
        n_batches = 0

        for batch in loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            metrics = self.criterion(pred, y)
            metrics['loss'].backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += metrics['loss'].item()
            total_factor += metrics['factor']
            n_batches += 1

        return {'loss': total_loss / n_batches, 'factor': total_factor / n_batches}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_factor = 0.0
        n_batches = 0

        for batch in loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            pred = self.model(x)
            metrics = self.criterion(pred, y)
            total_loss += metrics['loss'].item()
            total_factor += metrics['factor']
            n_batches += 1

        return {'loss': total_loss / n_batches, 'factor': total_factor / n_batches}

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, self.checkpoint_dir / filename)
