"""Closed-loop ID/FD horizon rollout with history cloning."""
import torch


def closed_loop_idfd_rollout(
    delta_x_des_H, x0, xd0,
    horizon_steps, id_model, X_mean, X_std,
    sequence_fd, sequence_id,
    model_dt=0.05,
):
    """Run H-step ID/FD rollout, returns accumulated delta_fd_H.

    Clones and restores GRU histories. Does not mutate live state.
    """
    id_hist = sequence_id._step_history.detach().clone()
    fd_hist = sequence_fd._step_history.detach().clone()
    x_r = x0.detach().clone()
    xd_r = xd0.detach().clone()
    fd_sum = torch.zeros_like(delta_x_des_H)
    device = delta_x_des_H.device

    B = delta_x_des_H.shape[0]
    for _ in range(horizon_steps):
        dx_step = delta_x_des_H / horizon_steps
        v_step = dx_step / model_dt
        id_in = torch.cat([dx_step, v_step, x_r, xd_r], dim=-1)
        with torch.no_grad():
            valve = id_model((id_in - X_mean) / X_std)
        sequence_fd.push_current_state(valve, x_r, xd_r)
        fd_d = sequence_fd.predict_delta_extension().delta_extension_raw
        fd_sum += fd_d
        x_r += fd_d
        xd_r[:, :] = v_step

    sequence_id._step_history = id_hist
    sequence_fd._step_history = fd_hist
    return fd_sum
