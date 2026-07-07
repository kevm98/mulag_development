"""
Simple Analytical Task-to-Cylinder Adapter.

Instead of learning, uses a simple kinematic approximation:
- Assumes differential cylinder arrangement
- Y motion: cylinders 1 vs 2
- Z motion: cylinders 3 vs 4

This allows immediate testing of the pipeline while learned adapter is being fixed.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class AnalyticalAdapter(nn.Module):
    """Analytical approximation of task-to-cylinder mapping."""

    def __init__(
        self,
        y_scale: float = 2.0,
        z_scale: float = 2.0,
        max_delta_cylinder: float = 0.0025,
    ):
        """
        Args:
            y_scale: Kinematic ratio for Y motion (EE_m to cylinder_m)
            z_scale: Kinematic ratio for Z motion
            max_delta_cylinder: Maximum cylinder command magnitude (safety)
        """
        super().__init__()

        self.y_scale = y_scale
        self.z_scale = z_scale
        self.max_delta_cylinder = max_delta_cylinder

        # For compatibility with learned adapter interface
        self.register_buffer("cylinder_min", torch.tensor([-0.2, -0.2, -0.2, -0.2]))
        self.register_buffer("cylinder_max", torch.tensor([0.2, 0.2, 0.2, 0.2]))

        print(f"[AnalyticalAdapter] Using simple Jacobian approximation")
        print(f"  Y scale: {y_scale}, Z scale: {z_scale}")
        print(f"  Max cylinder delta: {max_delta_cylinder*1000}mm")

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
        Compute cylinder commands from desired EE motion.

        Args:
            desired_delta_ee_yz: (B, 2) Desired EE motion [dy, dz]
            current_ee_yz: (B, 2) Current EE position [y, z] (unused for analytical)
            cylinder_pos: (B, 4) Current cylinder positions
            cylinder_vel: (B, 4) Current cylinder velocities (unused)
            previous_cylinder_delta: (B, 4) Previous commands (unused)
            enforce_limits: Whether to clip outputs to safe range

        Returns:
            cylinder_delta: (B, 4) Commanded cylinder increments
        """
        B = desired_delta_ee_yz.shape[0]
        device = desired_delta_ee_yz.device

        dy = desired_delta_ee_yz[:, 0]
        dz = desired_delta_ee_yz[:, 1]

        # Differential cylinder control
        # Cylinder 1 & 2: Y motion (positive = extend cyl1, retract cyl2)
        # Cylinder 3 & 4: Z motion (positive = extend cyl3, retract cyl4)

        cylinder_delta = torch.zeros((B, 4), device=device)
        cylinder_delta[:, 0] = dy * self.y_scale / 2   # cyl1: +Y
        cylinder_delta[:, 1] = -dy * self.y_scale / 2  # cyl2: -Y (opposite)
        cylinder_delta[:, 2] = dz * self.z_scale / 2   # cyl3: +Z
        cylinder_delta[:, 3] = -dz * self.z_scale / 2  # cyl4: -Z (opposite)

        # Clip to max magnitude
        cylinder_delta = torch.clamp(
            cylinder_delta,
            -self.max_delta_cylinder,
            self.max_delta_cylinder
        )

        # Safety: check cylinder limits
        if enforce_limits:
            next_cylinder_pos = cylinder_pos + cylinder_delta
            margin = 0.01  # 10mm safety margin

            # Clip to prevent hitting limits
            cylinder_delta = torch.clamp(
                cylinder_delta,
                self.cylinder_min + margin - cylinder_pos,
                self.cylinder_max - margin - cylinder_pos,
            )

        return cylinder_delta

    def set_cylinder_limits(self, cylinder_min: torch.Tensor, cylinder_max: torch.Tensor):
        """Set cylinder position limits for safety."""
        self.cylinder_min = cylinder_min
        self.cylinder_max = cylinder_max

    def save(self, path: Path | str, include_stats: bool = True):
        """Save configuration (for compatibility with learned adapter)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "type": "analytical",
            "config": {
                "y_scale": self.y_scale,
                "z_scale": self.z_scale,
                "max_delta_cylinder": self.max_delta_cylinder,
            },
        }

        torch.save(checkpoint, path)
        print(f"[AnalyticalAdapter] Saved to {path}")

    @classmethod
    def load(cls, path: Path | str, device: str = "cuda:0") -> "AnalyticalAdapter":
        """Load analytical adapter from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)

        if checkpoint.get("type") != "analytical":
            raise ValueError(f"Checkpoint is not an analytical adapter: {path}")

        config = checkpoint["config"]
        model = cls(**config).to(device)

        print(f"[AnalyticalAdapter] Loaded from {path}")
        return model


def create_analytical_adapter(
    y_scale: float = 2.0,
    z_scale: float = 2.0,
    max_delta_cylinder: float = 0.0025,
    output_path: Optional[str] = None,
) -> AnalyticalAdapter:
    """
    Create and optionally save an analytical adapter.

    Args:
        y_scale: Kinematic ratio for Y axis
        z_scale: Kinematic ratio for Z axis
        max_delta_cylinder: Max cylinder command per step
        output_path: Optional path to save adapter

    Returns:
        AnalyticalAdapter instance
    """
    adapter = AnalyticalAdapter(
        y_scale=y_scale,
        z_scale=z_scale,
        max_delta_cylinder=max_delta_cylinder,
    )

    # Set reasonable cylinder limits
    adapter.set_cylinder_limits(
        torch.tensor([-0.2, -0.2, -0.2, -0.2]),
        torch.tensor([0.2, 0.2, 0.2, 0.2]),
    )

    if output_path:
        adapter.save(output_path)

    return adapter


if __name__ == "__main__":
    """Test analytical adapter with simple inputs."""
    print("="*80)
    print("ANALYTICAL ADAPTER TEST")
    print("="*80)

    adapter = create_analytical_adapter(
        y_scale=2.0,
        z_scale=2.0,
        max_delta_cylinder=0.0025,
    )

    # Test inputs
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    adapter = adapter.to(device)

    # Create test batch (1 sample)
    desired_ee = torch.tensor([[0.005, 0.0]], device=device)  # +5mm Y
    current_ee = torch.zeros((1, 2), device=device)
    cyl_pos = torch.zeros((1, 4), device=device)
    cyl_vel = torch.zeros((1, 4), device=device)
    prev_delta = torch.zeros((1, 4), device=device)

    print("\nTest: Desired +5mm Y motion")
    output = adapter(desired_ee, current_ee, cyl_pos, cyl_vel, prev_delta)
    print(f"  Cylinder deltas (mm): {output[0].cpu().numpy() * 1000}")
    print(f"  Expected: [+2.5, -2.5, 0, 0] (max clipped)")

    print("\nTest: Desired +5mm Z motion")
    desired_ee = torch.tensor([[0.0, 0.005]], device=device)
    output = adapter(desired_ee, current_ee, cyl_pos, cyl_vel, prev_delta)
    print(f"  Cylinder deltas (mm): {output[0].cpu().numpy() * 1000}")
    print(f"  Expected: [0, 0, +2.5, -2.5] (max clipped)")

    print("\nTest: Desired +5mm Y, +5mm Z (diagonal)")
    desired_ee = torch.tensor([[0.005, 0.005]], device=device)
    output = adapter(desired_ee, current_ee, cyl_pos, cyl_vel, prev_delta)
    print(f"  Cylinder deltas (mm): {output[0].cpu().numpy() * 1000}")
    print(f"  Expected: [+2.5, -2.5, +2.5, -2.5] (all max)")

    # Save for use in pipeline
    output_path = "outputs/ppo1_adapter/analytical_adapter.pt"
    adapter.save(output_path)
    print(f"\nSaved adapter to: {output_path}")
    print("="*80)
