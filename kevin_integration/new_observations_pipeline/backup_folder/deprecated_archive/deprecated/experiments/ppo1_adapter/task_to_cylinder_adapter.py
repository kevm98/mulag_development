"""
Task-to-Cylinder Adapter: Learned inverse model mapping EE commands to cylinder commands.

Input (16D):
    - desired_delta_ee_yz (2D): Desired end-effector motion [dy, dz]
    - current_ee_yz (2D): Current end-effector position
    - cylinder_pos (4D): Current cylinder extensions
    - cylinder_vel (4D): Current cylinder velocities
    - previous_cylinder_delta (4D): Previous commanded cylinder increment

Output (4D):
    - desired_cylinder_delta (4D): Commanded cylinder increment [Δcyl₁, Δcyl₂, Δcyl₃, Δcyl₄]

Purpose:
    - Bridge between task-space PPO commands and cylinder-space lower-level control
    - Learn configuration-dependent kinematics without analytical Jacobian
    - Enable hierarchical control architecture
"""

from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional


class TaskToCylinderAdapter(nn.Module):
    """MLP adapter mapping desired EE motion to cylinder commands."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        max_delta_cylinder: float = 0.0025,
    ):
        """
        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            use_layer_norm: Whether to use LayerNorm
            max_delta_cylinder: Maximum cylinder increment (for output scaling and safety)
        """
        super().__init__()

        self.input_dim = 16  # desired_ee(2) + ee(2) + cyl_pos(4) + cyl_vel(4) + prev_delta(4)
        self.output_dim = 4  # cylinder_delta
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.max_delta_cylinder = max_delta_cylinder

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

        # Cylinder limits for safety clipping (set from dataset)
        self.register_buffer("cylinder_min", torch.full((4,), -0.2))
        self.register_buffer("cylinder_max", torch.full((4,), 0.2))

    def forward(
        self,
        desired_delta_ee_yz: torch.Tensor,
        current_ee_yz: torch.Tensor,
        cylinder_pos: torch.Tensor,
        cylinder_vel: torch.Tensor,
        previous_cylinder_delta: torch.Tensor,
        enforce_limits: bool = True,
    ) -> torch.Tensor:
        """
        Predict cylinder command from desired end-effector motion.

        Args:
            desired_delta_ee_yz: (B, 2) Desired EE motion [dy, dz]
            current_ee_yz: (B, 2) Current EE position [y, z]
            cylinder_pos: (B, 4) Current cylinder extensions
            cylinder_vel: (B, 4) Current cylinder velocities
            previous_cylinder_delta: (B, 4) Previous commanded delta
            enforce_limits: Whether to clip output to safe range

        Returns:
            desired_cylinder_delta: (B, 4) Commanded cylinder increment
        """
        # Concatenate inputs
        x = torch.cat([
            desired_delta_ee_yz,
            current_ee_yz,
            cylinder_pos,
            cylinder_vel,
            previous_cylinder_delta,
        ], dim=-1)

        # Normalize input
        x = (x - self.input_mean) / (self.input_std + 1e-8)

        # Forward pass
        normalized_output = self.network(x)

        # Denormalize and scale output
        output = normalized_output * self.max_delta_cylinder

        # Safety clipping to prevent hitting cylinder limits
        if enforce_limits:
            next_cylinder_pos = cylinder_pos + output
            # Clip to safe range with margin
            margin = 0.01  # 10mm safety margin
            output = torch.clamp(
                output,
                self.cylinder_min + margin - cylinder_pos,
                self.cylinder_max - margin - cylinder_pos,
            )

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

    def set_cylinder_limits(self, cylinder_min: torch.Tensor, cylinder_max: torch.Tensor):
        """Set physical cylinder limits for safety clipping."""
        self.cylinder_min.copy_(cylinder_min)
        self.cylinder_max.copy_(cylinder_max)

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
                "max_delta_cylinder": self.max_delta_cylinder,
            },
        }

        if include_stats:
            checkpoint["normalization"] = {
                "input_mean": self.input_mean.cpu(),
                "input_std": self.input_std.cpu(),
                "output_mean": self.output_mean.cpu(),
                "output_std": self.output_std.cpu(),
            }
            checkpoint["limits"] = {
                "cylinder_min": self.cylinder_min.cpu(),
                "cylinder_max": self.cylinder_max.cpu(),
            }

        torch.save(checkpoint, path)
        print(f"[TaskToCylinderAdapter] Saved to {path}")

    @classmethod
    def load(cls, path: Path | str, device: str = "cuda:0") -> "TaskToCylinderAdapter":
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
                norm.get("output_mean", torch.zeros(4)).to(device),
                norm.get("output_std", torch.ones(4)).to(device),
            )

        # Load cylinder limits if available
        if "limits" in checkpoint:
            limits = checkpoint["limits"]
            model.set_cylinder_limits(
                limits["cylinder_min"].to(device),
                limits["cylinder_max"].to(device),
            )

        model.to(device)
        model.eval()
        print(f"[TaskToCylinderAdapter] Loaded from {path}")
        return model


def compute_adapter_supervised_loss(
    predicted_cylinder_delta: torch.Tensor,
    target_cylinder_delta: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss between predicted and target cylinder commands."""
    return nn.functional.mse_loss(predicted_cylinder_delta, target_cylinder_delta)


def compute_adapter_forward_consistency_loss(
    predicted_cylinder_delta: torch.Tensor,
    desired_delta_ee_yz: torch.Tensor,
    predicted_delta_ee_yz: torch.Tensor,
) -> torch.Tensor:
    """
    Compute forward-consistency loss.

    The forward model predicts what EE motion will result from the adapter's
    predicted cylinder command. This loss ensures the adapter output is
    physically consistent.
    """
    return nn.functional.mse_loss(predicted_delta_ee_yz, desired_delta_ee_yz)


def compute_adapter_command_regularization(
    predicted_cylinder_delta: torch.Tensor,
) -> torch.Tensor:
    """Encourage small, smooth commands."""
    return (predicted_cylinder_delta ** 2).mean()


def compute_adapter_limit_penalty(
    cylinder_pos: torch.Tensor,
    predicted_cylinder_delta: torch.Tensor,
    cylinder_min: torch.Tensor,
    cylinder_max: torch.Tensor,
    margin: float = 0.02,
) -> torch.Tensor:
    """
    Penalize commands that push cylinders near limits.

    Args:
        margin: Safety margin in meters (e.g., 0.02 = 20mm)
    """
    next_pos = cylinder_pos + predicted_cylinder_delta

    # Distance to lower limit
    dist_to_min = next_pos - (cylinder_min + margin)
    lower_violation = torch.relu(-dist_to_min)  # Positive when too close

    # Distance to upper limit
    dist_to_max = (cylinder_max - margin) - next_pos
    upper_violation = torch.relu(-dist_to_max)  # Positive when too close

    total_violation = lower_violation + upper_violation
    return total_violation.mean()


def compute_adapter_total_loss(
    adapter: TaskToCylinderAdapter,
    forward_model,
    desired_delta_ee_yz: torch.Tensor,
    current_ee_yz: torch.Tensor,
    cylinder_pos: torch.Tensor,
    cylinder_vel: torch.Tensor,
    previous_cylinder_delta: torch.Tensor,
    target_cylinder_delta: torch.Tensor,
    lambda_consistency: float = 1.0,
    lambda_command: float = 0.01,
    lambda_limit: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Compute total adapter training loss with all components.

    Returns dict with:
        - loss_total: Combined loss
        - loss_supervised: Supervised MSE loss
        - loss_consistency: Forward-consistency loss
        - loss_command: Command regularization
        - loss_limit: Limit penalty
    """
    # Adapter prediction
    predicted_cylinder_delta = adapter(
        desired_delta_ee_yz,
        current_ee_yz,
        cylinder_pos,
        cylinder_vel,
        previous_cylinder_delta,
        enforce_limits=False,  # Don't clip during training
    )

    # Supervised loss
    loss_supervised = compute_adapter_supervised_loss(
        predicted_cylinder_delta,
        target_cylinder_delta,
    )

    # Forward-consistency loss
    if forward_model is not None:
        predicted_delta_ee_yz = forward_model(
            current_ee_yz,
            cylinder_pos,
            cylinder_vel,
            previous_cylinder_delta,
            predicted_cylinder_delta,
        )
        loss_consistency = compute_adapter_forward_consistency_loss(
            predicted_cylinder_delta,
            desired_delta_ee_yz,
            predicted_delta_ee_yz,
        )
    else:
        loss_consistency = torch.tensor(0.0, device=predicted_cylinder_delta.device)

    # Command regularization
    loss_command = compute_adapter_command_regularization(predicted_cylinder_delta)

    # Limit penalty
    loss_limit = compute_adapter_limit_penalty(
        cylinder_pos,
        predicted_cylinder_delta,
        adapter.cylinder_min,
        adapter.cylinder_max,
    )

    # Total loss
    loss_total = (
        loss_supervised
        + lambda_consistency * loss_consistency
        + lambda_command * loss_command
        + lambda_limit * loss_limit
    )

    return {
        "loss_total": loss_total,
        "loss_supervised": loss_supervised,
        "loss_consistency": loss_consistency,
        "loss_command": loss_command,
        "loss_limit": loss_limit,
    }
