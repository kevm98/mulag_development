# Active Pipeline Files

| Role | File |
|---|---|
| Task registration | `kevin_integration/tasks/__init__.py` |
| DirectRL environment | `kevin_integration/tasks/mulag_adaptive_rl_env.py` |
| Environment and PPO config | `kevin_integration/tasks/mulag_adaptive_rl_env_cfg.py` |
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
| Unit tests | `kevin_integration/rom_cylinders/test_*.py` |
| AAM/ID/SD/FD checkpoints | `kevin_integration/models/` |
| ROM data contract | `kevin_integration/rom_cylinders/ROM_DATA_CONTRACT.md` |
| Active control contract | `kevin_integration/ISAACLAB_CONTROL_CONTRACT.md` |
