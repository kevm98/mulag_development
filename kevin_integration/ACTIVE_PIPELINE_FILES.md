# Active Pipeline Files

| Role | File |
|---|---|
| Task registration | `kevin_integration/tasks/__init__.py` |
| DirectRL environment | `kevin_integration/tasks/mulag_adaptive_rl_env.py` |
| Environment and PPO config | `kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py` |
| RRLAB obstacle conventions | `kevin_integration/tasks/rrlab_reference_conventions.py` |
| Task-space reference contract | `kevin_integration/tasks/task_space_reference.py` |
| Learned checkpoint loaders | `kevin_integration/controllers/learned_models.py` |
| AAM/ID runtime controller | `kevin_integration/rl/adaptive_rl_controller.py` |
| Runtime feature assembly | `kevin_integration/rl/observation_builder.py` |
| Reward terms | `kevin_integration/rl/reward_terms.py` |
| Final ROM inference | `kevin_integration/rom_cylinders/valve_conditioned_causal_rom.py` |
| Multirate ROM wrapper | `kevin_integration/rom_cylinders/multirate_rom_wrapper.py` |
| Coordinate adapter | `kevin_integration/rom_cylinders/cylinder_coordinate_adapter.py` |
| Coordinate mapping data | `kevin_integration/rom_cylinders/cylinder_coordinate_mapping.json` |
| Active final ROM checkpoint package | `kevin_integration/rom_cylinders/exported_valve_conditioned_causal_rom/` |
| Play/demo entrypoint | `kevin_integration/scripts/play_mulag_adaptive_rl.py` |
| Train entrypoint | `kevin_integration/scripts/train_mulag_adaptive_rl.py` |
| Deterministic checkpoint evaluator | `kevin_integration/scripts/evaluate_mulag_checkpoints.py` |
| ROM unit tests | `kevin_integration/rom_cylinders/test_*.py` |
| Task-space/FD contract tests | `kevin_integration/tasks/test_task_space_reference.py` |
| RRLAB convention tests | `kevin_integration/tasks/test_rrlab_reference_conventions.py` |
| FD observer contract tests | `kevin_integration/tasks/test_fd_observer_contract.py` |
| AAM/ID/SD/FD checkpoints | `kevin_integration/models/` |
| ROM data contract | `kevin_integration/rom_cylinders/ROM_DATA_CONTRACT.md` |
| Active control contract | `kevin_integration/ISAACLAB_CONTROL_CONTRACT.md` |
| Full hydraulic pipeline | `kevin_integration/FULL_HYDRAULIC_PIPELINE.md` |
| Curriculum/learnability repair tests | `kevin_integration/tasks/test_curriculum_learnability_repair.py` |
| Fixed final-pose contracts | `kevin_integration/tasks/test_fixed_final_pose_task.py` |
| Play launcher contracts | `kevin_integration/tasks/test_play_launcher_contract.py` |
| Checkpoint evaluation contracts | `kevin_integration/tasks/test_checkpoint_evaluation.py` |
| Reward-balance contracts | `kevin_integration/tasks/test_reward_balance_repair.py` |
| Live AAM runtime wiring contracts | `kevin_integration/tasks/test_aam_runtime_wiring_contract.py` |
| Control architecture image | `kevin_integration/Control_Pipeline.png` |
| Editable control architecture | `kevin_integration/Control_Pipeline.svg` |

The canonical task-space configuration samples one fixed robot-base
`[x, y, z, qw, qx, qy, qz]` target per episode, uses quaternion-geodesic
orientation tracking, requires 50 settled control steps for success, and keeps
the new goal-difficulty curriculum disabled by default until runtime gates pass.
The active reward contract is `task_space_reward_balance_v2`. Isaac runtime
validation remains pending.

The active online adaptation branch is:

```text
q[5], qdot[5], dP[4], previous applied ValveCmd[4]
-> one 100-step AAM history
-> frozen aam.pth
-> z_arm_hat[24]
   |-> PPO observation[41:65]
   `-> matching ID conditioning input
```

`_prev_valve_cmd` is the established previous applied command field. AAM is
context-only and the offline System Factor Encoder is not used at runtime.
