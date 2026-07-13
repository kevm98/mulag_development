# DLS lower-level controller

This folder owns deterministic control only:

- `ee_cascade.py`: cutter-space damped-least-squares IK using the cached,
  physics-measured 3x4 Jacobian.
- `cylinder.py`: full coupled cylinder DLS closed on measured ram positions.
- `ground_paths.py`: physics-settled ground-path data access.
- `kinematics.py`: stepped-PhysX Jacobian measurement helpers.
- `tests/`: non-vacuous controller and trajectory gates.

RL code does not belong here. Do not add MPC or raw-cylinder policy actions. The primary
operational diagnostic is `tests/stop_and_mow_gate.py`; its truck relocation gate is
currently an open blocker documented in `../docs/stop_and_mow_report.md`.
