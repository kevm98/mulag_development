# Stage 2: prediction_only

**Purpose**: Quantify whether FD kills the command. ID feedforward + PID controls valve, FD predicts delta.

**Config**: servo_mode: `id_ff_plus_cylinder_velocity_pid`, fd_usage: `prediction_only`, kp=6.0/ki=0/kd=0

## Results

Identical to Stage 1 bypass (same RMSE, sign_acc, atten values).

ID feedforward produces near-zero valve_ff (Stage A). PID with kp=6 produces non-zero valve_pid, but FD(valve_total) produces: zero delta (when valve_total near zero from broken ID) OR bypass is still active.

## Verdict

**Stage A confirmed**: ID model does not respond to velocity-only commands. The model was trained on positional target deltas (`target_delta_x`) and produces valve commands proportional to these. When `target_delta_x` is tiny (v_des * dt ~ 0.000167m), ID outputs near-zero.

The FD model does NOT kill the command — the command never reaches FD because ID produces nothing.
