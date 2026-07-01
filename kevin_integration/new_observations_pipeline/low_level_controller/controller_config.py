"""Controller configuration constants.

Final selected: current_horizon_idfd_ff
    delta_cmd_H = fd_gain[c] * delta_fd_H
"""
CONTROLLER_CURRENT_HORIZON_IDFD_FF = "current_horizon_idfd_ff"
DEFAULT_FD_GAINS = [2.0, 3.0, 3.0, 3.0]
DEFAULT_HORIZON_STEPS = 10
DEFAULT_ACTION_SCALE_M = 0.01
TRACKING_GAIN = 0.0
DELTA_CLIP_M = 0.02
MODEL_DT = 0.05
