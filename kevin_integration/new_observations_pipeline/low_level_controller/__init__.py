"""Low-level horizon displacement controller: ID/FD rollout, fd_gain application.

Controller: current_horizon_idfd_ff
    delta_cmd_H = fd_gain[c] * delta_fd_H

Diagnostic only: position_ramp_pi, horizon_idfd_ff_pi
"""
from .controller_config import (
    DEFAULT_FD_GAINS, DEFAULT_HORIZON_STEPS, DEFAULT_ACTION_SCALE_M,
    CONTROLLER_CURRENT_HORIZON_IDFD_FF,
)
from .idfd_rollout import closed_loop_idfd_rollout
from .horizon_idfd_controller import compute_horizon_command
