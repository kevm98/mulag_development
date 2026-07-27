"""Runtime bundle for AAM SystemFactor: export and inference."""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict

from AAM_SystemFactor.models.aam_lstm import AAMLSTM
from AAM_SystemFactor.data.normalization import FeatureScaler, TargetScaler
from AAM_SystemFactor.data.factor_targets import TARGET_NAMES, NOMINAL_TARGET


def export_runtime_bundle(
    model: AAMLSTM,
    input_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    config: Dict,
    output_path: str,
) -> None:
    """Export a self-contained runtime checkpoint.

    Args:
        model: trained AAM LSTM
        input_scaler: fitted input scaler
        target_scaler: fitted target scaler
        config: metadata dict
        output_path: destination .pt file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'output_size': model.output_size,
        'input_scaler_mean': input_scaler.mean.squeeze().tolist(),
        'input_scaler_std': input_scaler.std.squeeze().tolist(),
        'target_scaler_mean': target_scaler.mean.squeeze().tolist(),
        'target_scaler_std': target_scaler.std.squeeze().tolist(),
        'feature_names': config.get('feature_names', []),
        'target_names': TARGET_NAMES,
        'T': config.get('T', 10),
        'version': '0.1.0',
    }
    torch.save(checkpoint, output_path)


class AAMPredictor:
    """Denormalized factor prediction from a runtime bundle."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = AAMLSTM(
            input_size=ckpt['input_size'],
            hidden_size=ckpt['hidden_size'],
            num_layers=ckpt['num_layers'],
            output_size=ckpt['output_size'],
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.device = device

        self.input_scaler = FeatureScaler()
        self.input_scaler.mean = np.array(ckpt['input_scaler_mean']).reshape(1, -1)
        self.input_scaler.std = np.array(ckpt['input_scaler_std']).reshape(1, -1)
        self.input_scaler._fitted = True

        self.target_scaler = TargetScaler()
        self.target_scaler.mean = np.array(ckpt['target_scaler_mean']).reshape(1, -1)
        self.target_scaler.std = np.array(ckpt['target_scaler_std']).reshape(1, -1)
        self.target_scaler._fitted = True

        self.T = ckpt.get('T', 10)
        self.target_names = ckpt.get('target_names', TARGET_NAMES)

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict denormalized physical factors.

        Args:
            history: (B, T, 20) raw input features

        Returns:
            predicted_factors: (B, 7) denormalized physical factors
                in order: e_comp, e_flow, e_pressure, e_seal_1..4
        """
        assert history.ndim == 3, f"Expected (B, T, 20), got {history.shape}"
        assert history.shape[1] == self.T, f"Expected T={self.T}"
        assert history.shape[2] == 20, f"Expected 20 features"

        # Reshape for scaler: (B*T, 20) -> scale -> (B, T, 20)
        B, T, F = history.shape
        flat = history.reshape(B * T, F)
        scaled_flat = self.input_scaler.transform(flat)
        scaled = scaled_flat.reshape(B, T, F)

        with torch.no_grad():
            x = torch.tensor(scaled, dtype=torch.float32, device=self.device)
            pred_scaled = self.model(x).cpu().numpy()

        pred_denorm = self.target_scaler.inverse_transform(pred_scaled)
        return pred_denorm
