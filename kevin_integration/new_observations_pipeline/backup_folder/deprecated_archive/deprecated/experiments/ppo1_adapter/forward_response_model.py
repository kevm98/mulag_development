"""
Forward Response Model: Predicts end-effector motion from cylinder commands.

Input (18D):
    - current_ee_yz (2D): Current end-effector position
    - cylinder_pos (4D): Current cylinder extensions
    - cylinder_vel (4D): Current cylinder velocities
    - previous_cylinder_delta (4D): Previous commanded cylinder increment
    - candidate_cylinder_delta (4D): Candidate cylinder command

Output (2D):
    - predicted_delta_ee_y (1D): Predicted Y motion
    - predicted_delta_ee_z (1D): Predicted Z motion

Purpose:
    - Validate learned adapter outputs before execution
    - Provide forward-consistency loss for adapter training
    - Enable offline simulation and safety checks
"""

from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional


class ForwardResponseModel(nn.Module):
    """MLP forward model mapping cylinder commands to end-effector motion."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        max_delta_ee: float = 0.05,
    ):
        """
        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            use_layer_norm: Whether to use LayerNorm
            max_delta_ee: Maximum predicted EE motion (for output scaling)
        """
        super().__init__()

        self.input_dim = 18  # ee(2) + cyl_pos(4) + cyl_vel(4) + prev_delta(4) + cmd(4)
        self.output_dim = 2  # predicted delta_ee_yz
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.max_delta_ee = max_delta_ee

        # Build MLP
        layers = []
        in_dim = self.input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, self.output_dim))
        layers.append(nn.Tanh())  # Bounded output

        self.network = nn.Sequential(*layers)

        # Input/output statistics for normalization
        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_std", torch.ones(self.input_dim))
        self.register_buffer("output_mean", torch.zeros(self.output_dim))
        self.register_buffer("output_std", torch.ones(self.output_dim))

    def forward(
        self,
        ee_yz: torch.Tensor,
        cylinder_pos: torch.Tensor,
        cylinder_vel: torch.Tensor,
        previous_cylinder_delta: torch.Tensor,
        candidate_cylinder_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict end-effector motion from cylinder command.

        Args:
            ee_yz: (B, 2) Current EE position [y, z]
            cylinder_pos: (B, 4) Current cylinder extensions
            cylinder_vel: (B, 4) Current cylinder velocities
            previous_cylinder_delta: (B, 4) Previous commanded delta
            candidate_cylinder_delta: (B, 4) Candidate cylinder command

        Returns:
            predicted_delta_ee_yz: (B, 2) Predicted EE motion [dy, dz]
        """
        # Concatenate inputs
        x = torch.cat([
            ee_yz,
            cylinder_pos,
            cylinder_vel,
            previous_cylinder_delta,
            candidate_cylinder_delta,
        ], dim=-1)

        # Normalize input
        x = (x - self.input_mean) / (self.input_std + 1e-8)

        # Forward pass
        normalized_output = self.network(x)

        # Denormalize and scale output
        output = normalized_output * self.max_delta_ee

        return output

    def set_normalization_stats(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: Optional[torch.Tensor] = None,
        output_std: Optional[torch.Tensor] = None,
    ):
        """Set normalization statistics from dataset."""
        self.input_mean.copy_(input_mean)
        self.input_std.copy_(input_std)
        if output_mean is not None:
            self.output_mean.copy_(output_mean)
        if output_std is not None:
            self.output_std.copy_(output_std)

    def save(self, path: Path | str, include_stats: bool = True):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "use_layer_norm": self.use_layer_norm,
                "max_delta_ee": self.max_delta_ee,
            },
        }

        if include_stats:
            checkpoint["normalization"] = {
                "input_mean": self.input_mean.cpu(),
                "input_std": self.input_std.cpu(),
                "output_mean": self.output_mean.cpu(),
                "output_std": self.output_std.cpu(),
            }

        torch.save(checkpoint, path)
        print(f"[ForwardResponseModel] Saved to {path}")

    @classmethod
    def load(cls, path: Path | str, device: str = "cuda:0") -> "ForwardResponseModel":
        """Load model from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)

        # Create model from config
        config = checkpoint["config"]
        model = cls(**config)

        # Load weights
        model.load_state_dict(checkpoint["state_dict"])

        # Load normalization stats if available
        if "normalization" in checkpoint:
            norm = checkpoint["normalization"]
            model.set_normalization_stats(
                norm["input_mean"].to(device),
                norm["input_std"].to(device),
                norm.get("output_mean", torch.zeros(2)).to(device),
                norm.get("output_std", torch.ones(2)).to(device),
            )

        model.to(device)
        model.eval()
        print(f"[ForwardResponseModel] Loaded from {path}")
        return model


def compute_forward_model_metrics(
    model: ForwardResponseModel,
    ee_yz: torch.Tensor,
    cylinder_pos: torch.Tensor,
    cylinder_vel: torch.Tensor,
    previous_cylinder_delta: torch.Tensor,
    candidate_cylinder_delta: torch.Tensor,
    measured_delta_ee_yz: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute validation metrics for forward model.

    Returns dict with:
        - mae_y, mae_z: Mean absolute error per axis
        - rmse_y, rmse_z: Root mean squared error per axis
        - sign_accuracy_y, sign_accuracy_z: Fraction of correct signs
        - r2_y, r2_z: R-squared score per axis
    """
    model.eval()
    with torch.no_grad():
        predicted = model(
            ee_yz,
            cylinder_pos,
            cylinder_vel,
            previous_cylinder_delta,
            candidate_cylinder_delta,
        )

    # Per-axis errors
    error = predicted - measured_delta_ee_yz
    mae_y = error[:, 0].abs().mean().item()
    mae_z = error[:, 1].abs().mean().item()

    rmse_y = (error[:, 0] ** 2).mean().sqrt().item()
    rmse_z = (error[:, 1] ** 2).mean().sqrt().item()

    # Sign accuracy
    sign_pred = torch.sign(predicted)
    sign_true = torch.sign(measured_delta_ee_yz)
    sign_acc_y = (sign_pred[:, 0] == sign_true[:, 0]).float().mean().item()
    sign_acc_z = (sign_pred[:, 1] == sign_true[:, 1]).float().mean().item()

    # R² score
    ss_res_y = (error[:, 0] ** 2).sum()
    ss_tot_y = ((measured_delta_ee_yz[:, 0] - measured_delta_ee_yz[:, 0].mean()) ** 2).sum()
    r2_y = (1 - ss_res_y / (ss_tot_y + 1e-8)).item()

    ss_res_z = (error[:, 1] ** 2).sum()
    ss_tot_z = ((measured_delta_ee_yz[:, 1] - measured_delta_ee_yz[:, 1].mean()) ** 2).sum()
    r2_z = (1 - ss_res_z / (ss_tot_z + 1e-8)).item()

    return {
        "mae_y": mae_y,
        "mae_z": mae_z,
        "rmse_y": rmse_y,
        "rmse_z": rmse_z,
        "sign_accuracy_y": sign_acc_y,
        "sign_accuracy_z": sign_acc_z,
        "r2_y": r2_y,
        "r2_z": r2_z,
        "mae_mean": (mae_y + mae_z) / 2,
    }
