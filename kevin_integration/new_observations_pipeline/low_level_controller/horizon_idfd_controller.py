"""Horizon ID/FD displacement controller."""
import torch
from .controller_config import DEFAULT_FD_GAINS, DELTA_CLIP_M


def compute_horizon_command(delta_fd_H, fd_gains=None, delta_clip_m=None):
    """Compute deployment command: delta_cmd_H = fd_gain[c] * delta_fd_H."""
    if fd_gains is None:
        fd_gains = DEFAULT_FD_GAINS
    if delta_clip_m is None:
        delta_clip_m = DELTA_CLIP_M
    fd_gains_t = torch.tensor(fd_gains, dtype=torch.float32,
                               device=delta_fd_H.device).reshape(1, 4)
    delta_cmd_H = fd_gains_t * delta_fd_H
    delta_cmd_H = torch.clamp(delta_cmd_H, -delta_clip_m, delta_clip_m)
    return delta_cmd_H
