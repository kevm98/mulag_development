# Reset Pose Hold Report

- duration_s: 1.0
- horizon_steps: 10
- horizon_steps_executed: 6
- max_ee_drift_m: 0.277092
- max_tcp_base_frame_drift_m: 0.059856
- start_tcp_base_min: [-4e-06, -3.423824, -0.117758]
- start_tcp_base_max: [-4e-06, -3.423824, -0.117758]
- end_tcp_base_min: [-2e-06, -3.376653, -0.080912]
- end_tcp_base_max: [-2e-06, -3.376653, -0.080912]
- max_hold_target_delta_rad_or_m: 0.000000
- max_guard_fraction: 0.000000
- cylinder_targets_finite: True
- cylinder_targets_in_soft_limits: True
- cab_proxy_clear: True
- base_terminated_any: False
- base_truncated_any: False
- pass: False

Collision note: this headless script checks the reset TCP against the configured cab box
and verifies finite in-limit cylinder targets. No dedicated arm/truck pair contact sensor
is configured in PPO1, so pairwise collision is audited by the companion geometric script.
