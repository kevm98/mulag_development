"""Data normalization: training-only fit, JSON persistence."""

import json
import numpy as np
from pathlib import Path
from typing import Optional


class FeatureScaler:
    """Standardize features: (x - mean) / std."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray, epsilon: float = 1e-8) -> 'FeatureScaler':
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std = np.maximum(self.std, epsilon)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted")
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted")
        return X * self.std + self.mean

    def fit_transform(self, X: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        self.fit(X, epsilon)
        return self.transform(X)

    def save(self, path: str) -> None:
        data = {'mean': self.mean.squeeze().tolist(), 'std': self.std.squeeze().tolist()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        scaler = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        scaler.mean = np.array(data['mean']).reshape(1, -1)
        scaler.std = np.array(data['std']).reshape(1, -1)
        scaler._fitted = True
        return scaler

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class TargetScaler(FeatureScaler):
    """Scaler for 7-D targets. Uses standardization by default."""
    pass
