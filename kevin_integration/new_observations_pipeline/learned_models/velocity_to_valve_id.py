"""Fine-tuned ID model: maps (desired_velocity, state) -> valve_command."""
import torch
import numpy as np
from pathlib import Path


class VelocityToValveMLP(torch.nn.Module):
    """MLP: 16 -> 128 -> 128 -> 64 -> 4.

    Input: [desired_delta(4), desired_vel(4), ext(4), extvel(4)]
    Output: valve_cmd[4]
    """
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 4),
        )
    def forward(self, x):
        return self.net(x)


def load_finetuned_id_model(checkpoint_path: str | Path, device: str = "cuda:0"):
    """Load fine-tuned ID model and scalers from checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = VelocityToValveMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    X_mean = torch.from_numpy(ckpt["X_mean"].astype("float32")).to(device)
    X_std = torch.from_numpy(ckpt["X_std"].astype("float32")).to(device)
    return model, X_mean, X_std
