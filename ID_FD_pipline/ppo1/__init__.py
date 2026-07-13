"""Two-dimensional Y/Z residual PPO layered over the verified DLS solver."""

from .observations import EE_RESIDUAL_OBS_DIM, build_residual_observation
from .rewards import compute_residual_reward

__all__ = ["EE_RESIDUAL_OBS_DIM", "build_residual_observation", "compute_residual_reward"]
