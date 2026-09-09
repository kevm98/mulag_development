# Collision Audit Report

- audit_type: headless reset geometric proxy
- tcp_base_min: [0.0, -3.311152, -0.253226]
- tcp_base_max: [0.0, -3.311152, -0.253226]
- cab_proxy_clear: True
- cylinder_targets_finite: True
- cylinder_targets_in_soft_limits: True
- root_z_min_m: 1.764604
- root_z_ok: True
- max_guard_fraction: 0.000000
- pass: True

No PPO1 arm/truck pair contact sensor is currently configured, so this audit does
not claim pairwise contact-force separation. It verifies the reset TCP is outside
the configured cab exclusion box, the base is not below the ground proxy, and
persistent cylinder targets are finite and inside soft limits.
