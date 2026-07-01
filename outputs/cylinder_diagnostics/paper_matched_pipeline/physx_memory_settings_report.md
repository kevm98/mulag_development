# PPO1 PhysX Memory Settings Report

1. Existing task memory settings found:
   - `extensions/rrlab_tasks/rrlab_tasks/manager_based/manipulation/obstacle_avoidance/obstacle_avoidance_env_cfg.py` uses:
     - `gpu_max_rigid_contact_count = 2**16`
     - `gpu_found_lost_pairs_capacity = 2**16`
     - `gpu_total_aggregate_pairs_capacity = 2**16`
     - `gpu_collision_stack_size = 2**20`
     - `gpu_heap_capacity = 2**22`
     - `gpu_temp_buffer_capacity = 2**20`
   - `kevin_integration/utils/sim_memory.py` centralizes the same settings in `apply_kevin_sim_memory_optimizations`.

2. PPO1 memory settings before fix:
   - `MulagUpperLevelEnvCfg.__post_init__` already called `apply_kevin_sim_memory_optimizations(self.sim)`.
   - `HorizonIDFDEnv.__init__` already called `apply_kevin_sim_memory_optimizations(base_cfg.sim)`.
   - `train_upper_level_ppo_horizon_idfd.py` did not explicitly call the optimizer before creating `HorizonIDFDEnv`.

3. PPO1 memory settings after fix:
   - `MulagUpperLevelEnvCfg.__post_init__` still applies the shared optimizer.
   - `train_upper_level_ppo_horizon_idfd.py` now explicitly applies `apply_kevin_sim_memory_optimizations(env_cfg.sim)` before env construction.
   - `HorizonIDFDEnv.__init__` still applies the shared optimizer before constructing `MulagUpperLevelEnv`.
   - Effective shared settings are:
     - `render_interval = 2`
     - `gpu_max_rigid_contact_count = 2**16`
     - `gpu_found_lost_pairs_capacity = 2**16`
     - `gpu_total_aggregate_pairs_capacity = 2**16`
     - `gpu_collision_stack_size = 2**20`
     - `gpu_heap_capacity = 2**22`
     - `gpu_temp_buffer_capacity = 2**20`

4. Smoke-test command used:

```bash
./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 1 \
  --max_iterations 1 \
  --headless \
  --device cuda:0 \
  --horizon_steps 10 \
  --horizon_action_scale_m 0.01 \
  --fd_horizon_gain_per_cylinder "2.0,3.0,3.0,3.0"
```
