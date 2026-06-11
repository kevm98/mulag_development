# Debug Cleanup Report

Date: 2026-06-11

Status: active runtime/debug tree reviewed; no checkpoints, datasets, ROM exports,
FD scalers, coordinate evidence, or active tests were removed.

Reason: the final task-space architecture does not use an IK/reference bridge.
PPO is the `ee_pose -> Action[4]` mapper, and the active cleanup keeps scripts
that still provide unique Isaac runtime diagnostics for AAM/ID/ROM/FD behavior.
Obsolete IK blocker language has been removed from active Kevin control docs and
tests now scan the active task/play/train sources for stale IK assumptions.

Reference scan candidates under `kevin_integration/scripts/`:

- `evaluate_adaptive_rl_debug.py`
- `test_direct_joint_effort_authority.py`
- `test_final_causal_rom_integration.py`
- `test_joint1_authority.py`
- `test_mulag_torques.py`
- `test_reduced_hydraulic_rom_offline.py`
- `test_rom_position_target_integration.py`
- `visualize_joint_tracking_live.py`

No files were moved to `kevin_integration/archive/debug_legacy/`.

Cleanup can be revisited after:

1. `joint_space_validation` is user-run in Isaac and shows expected
   AAM/ID/ROM/FD telemetry.
2. `rl_task_space_tracking` PPO smoke training passes preflight and creates a
   usable checkpoint.
3. Trained-policy playback shows measured `Messerkopf` motion responding to
   task-space commands with acceptable FD load and cylinder clamp telemetry.
