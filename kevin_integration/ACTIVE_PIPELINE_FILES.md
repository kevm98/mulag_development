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
| Static/runtime reports | `kevin_integration/results/` |
| ROM unit tests | `kevin_integration/rom_cylinders/test_*.py` |
| Task-space/FD contract tests | `kevin_integration/tasks/test_task_space_reference.py` |
| RRLAB convention tests | `kevin_integration/tasks/test_rrlab_reference_conventions.py` |
| FD observer contract tests | `kevin_integration/tasks/test_fd_observer_contract.py` |
| AAM/ID/SD/FD checkpoints | `kevin_integration/models/` |
| ROM data contract | `kevin_integration/rom_cylinders/ROM_DATA_CONTRACT.md` |
| Active control contract | `kevin_integration/ISAACLAB_CONTROL_CONTRACT.md` |
| Full hydraulic pipeline | `kevin_integration/FULL_HYDRAULIC_PIPELINE.md` |
| Vehicle/reference audit | `kevin_integration/RRLAB_VEHICLE_AND_REFERENCE_AUDIT.md` |
| RRLAB task-space trace | `kevin_integration/RRLAB_TASKSPACE_REFERENCE_TRACE.md` |
| PPO/reference alignment note | `kevin_integration/PPO_ACTION_REFERENCE_ALIGNMENT.md` |
| Debug cleanup report | `kevin_integration/DEBUG_CLEANUP_REPORT.md` |
