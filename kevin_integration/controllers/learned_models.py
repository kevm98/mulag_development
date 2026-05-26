from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to use kevin_integration learned controllers. "
            "Install it in your IsaacLab/IsaacSim python environment."
        ) from exc


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    state_dict: Mapping[str, Any]
    meta: Dict[str, Any]


def _split_checkpoint(obj: Any, path: Path) -> LoadedCheckpoint:
    # Two formats exist in this repo:
    # 1) full checkpoint dict with 'model_state_dict'
    # 2) raw state_dict, OrderedDict-like
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
        meta = {k: v for k, v in obj.items() if k != "model_state_dict"}
        return LoadedCheckpoint(path=path, state_dict=state_dict, meta=meta)

    if isinstance(obj, dict):
        # assume it is already a state_dict
        return LoadedCheckpoint(path=path, state_dict=obj, meta={})

    raise TypeError(f"Unsupported checkpoint type {type(obj)} at {path}")


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> LoadedCheckpoint:
    """Loads a .pth file produced by torch.save.

    Security note: torch.load unpickles Python objects and is unsafe on untrusted files.
    """
    torch = _require_torch()
    p = Path(path)

    try:
        obj = torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        obj = torch.load(p, map_location=map_location)

    return _split_checkpoint(obj, p)


def infer_lstm_dims_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    prefix: str,
) -> Tuple[int, int, int]:
    """Infer (input_dim, hidden_dim, num_layers) from LSTM weights in a state_dict."""
    key0 = f"{prefix}.weight_ih_l0"

    if key0 not in state_dict:
        raise KeyError(f"Missing '{key0}' in state_dict")

    w0 = state_dict[key0]

    if not hasattr(w0, "shape"):
        raise TypeError(f"Expected tensor for '{key0}', got {type(w0)}")

    input_dim = int(w0.shape[1])
    gate_dim = int(w0.shape[0])

    if gate_dim % 4 != 0:
        raise ValueError(f"Unexpected LSTM gate dim for '{key0}': {tuple(w0.shape)}")

    hidden_dim = gate_dim // 4

    # Count layers by looking for weight_ih_l{k}
    num_layers = 0
    while True:
        k = f"{prefix}.weight_ih_l{num_layers}"
        if k in state_dict:
            num_layers += 1
            continue
        break

    if num_layers <= 0:
        raise ValueError(f"Could not infer num_layers from prefix '{prefix}'")

    return input_dim, hidden_dim, num_layers


def _activation(name: str):
    torch = _require_torch()
    nn = torch.nn

    n = name.lower()

    if n == "relu":
        return nn.ReLU()
    if n == "elu":
        return nn.ELU()
    if n == "tanh":
        return nn.Tanh()
    if n == "gelu":
        return nn.GELU()
    if n == "leakyrelu":
        return nn.LeakyReLU()
    if n == "identity":
        return nn.Identity()

    raise ValueError(f"Unknown activation '{name}'")


def _read_int_meta(meta: Mapping[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        if key not in meta:
            continue

        value = meta[key]

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def _fix_feature_normalizer(vec, *, expected_dim: int, model_name: str, normalizer_name: str):
    """Fix checkpoint normalizer size when metadata stores dynamic + context features.

    Runtime ZArm models receive only dynamic features.
    The context part is replaced by z_arm_hat.
    """
    if vec is None:
        return None

    vec_len = len(vec)

    if vec_len == expected_dim:
        return vec

    if vec_len > expected_dim:
        print(
            f"[WARN][{model_name}] {normalizer_name} has {vec_len} values, "
            f"but {model_name} model expects {expected_dim}. "
            f"Slicing first {expected_dim} dynamic features."
        )
        return vec[:expected_dim]

    raise ValueError(
        f"[ERROR][{model_name}] {normalizer_name} has {vec_len} values, "
        f"but {model_name} model expects {expected_dim}."
    )


class AAMModel:
    """Arm Adaptation Module (AAM).

    Expects:
    - x_hist: (B, T, 18)

    Returns:
    - z_arm_hat: (B, 24)
    """

    def __init__(
        self,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        map_location: str = "cpu",
        device: str = "cpu",
        checkpoint_path: str | Path,
    ):
        torch = _require_torch()
        nn = torch.nn

        ckpt = load_checkpoint(checkpoint_path, map_location=map_location)

        in_dim, hidden_dim, num_layers = infer_lstm_dims_from_state_dict(
            ckpt.state_dict,
            prefix="temporal_encoder",
        )

        self.window_size = _read_int_meta(
            ckpt.meta,
            "history_len",
            "window_size",
            "aam_history_len",
        )

        # Infer head dims from linear weights
        head0_w = ckpt.state_dict["head.0.weight"]
        head3_w = ckpt.state_dict["head.3.weight"]

        head_dim = int(head0_w.shape[0])
        latent_dim = int(head3_w.shape[0])

        self.model = nn.Module()

        self.model.temporal_encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.model.head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(head_dim, latent_dim),
        )

        self.model.load_state_dict(ckpt.state_dict, strict=True)
        self.model.to(torch.device(device))
        self.model.eval()

    def __call__(self, x_hist):
        torch = _require_torch()

        with torch.no_grad():
            out, _ = self.model.temporal_encoder(x_hist)
            h_last = out[:, -1, :]
            return self.model.head(h_last)


class InverseDynamicsModel:
    """Inverse dynamics (ID) model.

    Expects:
    - x_hist: (B, T, 19)
      [q(5), qdot(5), qddot(5), dP(4)]
    - z_arm: (B, 24)

    Returns:
    - valve_cmd: (B, 4)
    """

    def __init__(
        self,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        map_location: str = "cpu",
        device: str = "cpu",
        checkpoint_path: str | Path,
    ):
        torch = _require_torch()
        nn = torch.nn

        ckpt = load_checkpoint(checkpoint_path, map_location=map_location)

        in_dim, hidden_dim, num_layers = infer_lstm_dims_from_state_dict(
            ckpt.state_dict,
            prefix="dynamic_encoder",
        )

        self.window_size = _read_int_meta(
            ckpt.meta,
            "history_len",
            "window_size",
            "id_window_size",
        )

        reg0_w = ckpt.state_dict["regressor.0.weight"]
        reg3_w = ckpt.state_dict["regressor.3.weight"]

        reg_hidden = int(reg0_w.shape[0])
        out_dim = int(reg3_w.shape[0])

        # Fusion dim is inferred from regressor input.
        fusion_dim = int(reg0_w.shape[1])
        z_dim = fusion_dim - hidden_dim

        if z_dim <= 0:
            raise ValueError(
                f"Could not infer z_dim from regressor input {fusion_dim} "
                f"and hidden_dim {hidden_dim}"
            )

        self.feature_mean = ckpt.meta.get("feat_mean", None)
        self.feature_std = ckpt.meta.get("feat_std", None)

        # ID runtime uses only dynamic features.
        # Some checkpoints store normalizers for:
        # 53 = 19 dynamic features + 34 context features.
        expected_dim = in_dim

        self.feature_mean = _fix_feature_normalizer(
            self.feature_mean,
            expected_dim=expected_dim,
            model_name="ID",
            normalizer_name="feat_mean",
        )

        self.feature_std = _fix_feature_normalizer(
            self.feature_std,
            expected_dim=expected_dim,
            model_name="ID",
            normalizer_name="feat_std",
        )

        self.model = nn.Module()

        self.model.dynamic_encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.model.regressor = nn.Sequential(
            nn.Linear(fusion_dim, reg_hidden),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(reg_hidden, out_dim),
            nn.Tanh(),
        )

        self.model.load_state_dict(ckpt.state_dict, strict=False)
        self.model.to(torch.device(device))
        self.model.eval()

    def normalize_features(self, x: Any):
        torch = _require_torch()

        if self.feature_mean is None or self.feature_std is None:
            return x

        mean = torch.as_tensor(
            self.feature_mean,
            dtype=torch.float32,
            device=x.device,
        ).reshape(1, 1, -1)

        std = torch.as_tensor(
            self.feature_std,
            dtype=torch.float32,
            device=x.device,
        ).reshape(1, 1, -1)

        if x.shape[-1] != mean.shape[-1]:
            raise RuntimeError(
                f"[ERROR][ID] Runtime feature dimension is {x.shape[-1]}, "
                f"but normalizer dimension is {mean.shape[-1]}. "
                f"Check ID feature assembly and checkpoint metadata."
            )

        return (x - mean) / (std + 1e-8)

    def __call__(self, x_hist, z_arm):
        torch = _require_torch()

        with torch.no_grad():
            x_hist = self.normalize_features(x_hist)
            out, _ = self.model.dynamic_encoder(x_hist)
            h_last = out[:, -1, :]
            fused = torch.cat([h_last, z_arm], dim=-1)
            return self.model.regressor(fused)


class SystemDynamicsModel:
    """System dynamics (SD) model.

    Expects:
    - x_hist: (B, T, 22)
      [q(5), qdot(5), dP(4), Fnet(4), valve_cmd(4)]
    - z_arm: (B, 24)

    Returns:
    - state prediction or state delta: (B, 18), depending on the checkpoint training convention
    """

    def __init__(
        self,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        map_location: str = "cpu",
        device: str = "cpu",
        checkpoint_path: str | Path,
    ):
        torch = _require_torch()
        nn = torch.nn

        ckpt = load_checkpoint(checkpoint_path, map_location=map_location)

        in_dim, hidden_dim, num_layers = infer_lstm_dims_from_state_dict(
            ckpt.state_dict,
            prefix="dynamic_encoder",
        )

        self.window_size = _read_int_meta(
            ckpt.meta,
            "history_len",
            "window_size",
            "sd_window_size",
        )

        reg0_w = ckpt.state_dict["regressor.0.weight"]
        reg3_w = ckpt.state_dict["regressor.3.weight"]

        reg_hidden = int(reg0_w.shape[0])
        out_dim = int(reg3_w.shape[0])

        fusion_dim = int(reg0_w.shape[1])
        z_dim = fusion_dim - hidden_dim

        if z_dim <= 0:
            raise ValueError(
                f"Could not infer z_dim from regressor input {fusion_dim} "
                f"and hidden_dim {hidden_dim}"
            )

        self.feature_mean = ckpt.meta.get("feat_mean", None)
        self.feature_std = ckpt.meta.get("feat_std", None)

        # SD runtime uses only dynamic features.
        # Some checkpoints store normalizers for:
        # 56 = 22 dynamic features + 34 context features.
        expected_dim = in_dim

        self.feature_mean = _fix_feature_normalizer(
            self.feature_mean,
            expected_dim=expected_dim,
            model_name="SD",
            normalizer_name="feat_mean",
        )

        self.feature_std = _fix_feature_normalizer(
            self.feature_std,
            expected_dim=expected_dim,
            model_name="SD",
            normalizer_name="feat_std",
        )

        self.model = nn.Module()

        self.model.dynamic_encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.model.regressor = nn.Sequential(
            nn.Linear(fusion_dim, reg_hidden),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(reg_hidden, out_dim),
        )

        self.model.load_state_dict(ckpt.state_dict, strict=False)
        self.model.to(torch.device(device))
        self.model.eval()

    def normalize_features(self, x: Any):
        torch = _require_torch()

        if self.feature_mean is None or self.feature_std is None:
            return x

        mean = torch.as_tensor(
            self.feature_mean,
            dtype=torch.float32,
            device=x.device,
        ).reshape(1, 1, -1)

        std = torch.as_tensor(
            self.feature_std,
            dtype=torch.float32,
            device=x.device,
        ).reshape(1, 1, -1)

        if x.shape[-1] != mean.shape[-1]:
            raise RuntimeError(
                f"[ERROR][SD] Runtime feature dimension is {x.shape[-1]}, "
                f"but normalizer dimension is {mean.shape[-1]}. "
                f"Check SD feature assembly and checkpoint metadata."
            )

        return (x - mean) / (std + 1e-8)

    def __call__(self, x_hist, z_arm):
        torch = _require_torch()

        with torch.no_grad():
            x_hist = self.normalize_features(x_hist)
            out, _ = self.model.dynamic_encoder(x_hist)
            h_last = out[:, -1, :]
            fused = torch.cat([h_last, z_arm], dim=-1)
            return self.model.regressor(fused)


class ForwardDynamicsModel:
    """Forward dynamics (FD) model.

    Expects:
    - x_hist: (B, T, 18)

    Returns:
    - torque: (B, 4)
    """

    def __init__(
        self,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        map_location: str = "cpu",
        device: str = "cpu",
        checkpoint_path: str | Path,
        scaler_path: str | Path,
    ):
        torch = _require_torch()
        nn = torch.nn

        ckpt = load_checkpoint(checkpoint_path, map_location=map_location)

        in_dim, hidden_dim, num_layers = infer_lstm_dims_from_state_dict(
            ckpt.state_dict,
            prefix="lstm",
        )

        self.window_size = _read_int_meta(
            ckpt.meta,
            "history_len",
            "window_size",
            "fd_window_size",
        )

        # Infer conv dims
        conv_w = ckpt.state_dict["hf_conv.0.weight"]
        conv_out = int(conv_w.shape[0])
        conv_in = int(conv_w.shape[1])
        kernel = int(conv_w.shape[2])

        # Infer skip dim
        skip_w = ckpt.state_dict["delta_skip.weight"]
        skip_dim = int(skip_w.shape[0])

        fc0_w = ckpt.state_dict["fc.0.weight"]
        fc_hidden = int(fc0_w.shape[0])
        fc_in = int(fc0_w.shape[1])
        out_dim = int(ckpt.state_dict["fc.3.weight"].shape[0])

        # Sanity check the concatenation sizes.
        if fc_in != hidden_dim + skip_dim + conv_out:
            raise ValueError(
                f"FD fc input dim mismatch: expected "
                f"{hidden_dim}+{skip_dim}+{conv_out}="
                f"{hidden_dim + skip_dim + conv_out}, got {fc_in}"
            )

        # Scaler
        scaler = load_checkpoint(scaler_path, map_location=map_location)

        self.x_mean = scaler.state_dict.get("x_mean")
        self.x_std = scaler.state_dict.get("x_std")
        self.y_mean = scaler.state_dict.get("y_mean")
        self.y_std = scaler.state_dict.get("y_std")

        self.model = nn.Module()

        self.model.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.model.delta_skip = nn.Linear(in_dim, skip_dim)

        self.model.hf_conv = nn.Sequential(
            nn.Conv1d(conv_in, conv_out, kernel_size=kernel),
            _activation(activation),
            nn.AdaptiveAvgPool1d(1),
        )

        self.model.fc = nn.Sequential(
            nn.Linear(fc_in, fc_hidden),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(fc_hidden, out_dim),
        )

        self.model.load_state_dict(ckpt.state_dict, strict=True)
        self.model.to(torch.device(device))
        self.model.eval()

    def normalize_x(self, x):
        torch = _require_torch()

        if self.x_mean is None or self.x_std is None:
            return x

        mean = torch.as_tensor(
            self.x_mean,
            dtype=torch.float32,
            device=x.device,
        ).reshape(1, 1, -1)

        std = torch.as_tensor(
            self.x_std,
            dtype=torch.float32,
            device=x.device,
        ).reshape(1, 1, -1)

        if x.shape[-1] != mean.shape[-1]:
            raise RuntimeError(
                f"[ERROR][FD] Runtime feature dimension is {x.shape[-1]}, "
                f"but scaler x_mean dimension is {mean.shape[-1]}. "
                f"Check FD feature assembly and fd_scaler.pth."
            )

        return (x - mean) / (std + 1e-8)

    def denormalize_y(self, y):
        torch = _require_torch()

        if self.y_mean is None or self.y_std is None:
            return y

        mean = torch.as_tensor(
            self.y_mean,
            dtype=torch.float32,
            device=y.device,
        ).reshape(1, -1)

        std = torch.as_tensor(
            self.y_std,
            dtype=torch.float32,
            device=y.device,
        ).reshape(1, -1)

        if y.shape[-1] != mean.shape[-1]:
            raise RuntimeError(
                f"[ERROR][FD] Runtime output dimension is {y.shape[-1]}, "
                f"but scaler y_mean dimension is {mean.shape[-1]}. "
                f"Check FD checkpoint and fd_scaler.pth."
            )

        return y * (std + 1e-8) + mean

    def __call__(self, x_hist):
        torch = _require_torch()

        with torch.no_grad():
            x_hist = self.normalize_x(x_hist)
            out, _ = self.model.lstm(x_hist)
            h_last = out[:, -1, :]
            skip = self.model.delta_skip(x_hist[:, -1, :])
            hf = self.model.hf_conv(x_hist.transpose(1, 2)).squeeze(-1)
            fused = torch.cat([h_last, skip, hf], dim=-1)
            y = self.model.fc(fused)
            return self.denormalize_y(y)
