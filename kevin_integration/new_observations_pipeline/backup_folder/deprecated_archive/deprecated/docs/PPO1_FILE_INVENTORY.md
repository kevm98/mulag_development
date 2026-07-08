# PPO1 Pipeline - File Inventory

## ✅ Working Pipeline (Use This!)

### 🎯 Entry Point
1. **kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py**
   - Main training script
   - Configures PPO, builds environment, runs training loop
   - Command: `./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py --ee_action_mode task_space_delta`

### 🎮 Core Pipeline (Execution Order)
2. **kevin_integration/new_observations_pipeline/ppo1/horizon_idfd_env_wrapper.py**
   - Main environment wrapper
   - Orchestrates entire pipeline
   - Integrates all components
   - **Key Classes:** `HorizonIDFDEnv`, `HorizonIDFDRslRlVecEnvWrapper`

3. **kevin_integration/new_observations_pipeline/low_level_controller/high_level_ee_controller.py**
   - Empirical Jacobian measurement
   - Task-space to cylinder-space mapping
   - **Key Class:** `HighLevelEEController`
   - **Key Method:** `measure_effective_jacobian()`, `map_ee_to_cylinder()`

4. **kevin_integration/new_observations_pipeline/low_level_controller/idfd_rollout.py**
   - Closed-loop multi-step prediction
   - **Key Function:** `closed_loop_idfd_rollout()`
   - Runs ID→FD→ID→FD loop for H steps

5. **kevin_integration/controllers/learned_models.py**
   - Kevin's ID/FD models (Simscape-trained)
   - **Line 245:** `InverseDynamicsModel` (desired_motion → valve_cmd)
   - **Line 554:** `ForwardDynamicsModel` (valve_cmd → predicted_motion)

### 🤖 Simulation Environment
6. **kevin_integration/new_observations_pipeline/ppo1/mulag_upper_level_env.py**
   - IsaacLab environment wrapper
   - Robot initialization, PhysX simulation interface
   - **Key Class:** `MulagUpperLevelEnv`

7. **kevin_integration/new_observations_pipeline/ppo1/mulag_upper_level_env_cfg.py**
   - Environment configuration
   - **Line 271:** `ee_action_mode = "task_space_delta"` ← Enable task-space!
   - **Line 67:** `task_space_ee_body_name = "Messerkopf"`

### 📊 Support Modules
8. **kevin_integration/new_observations_pipeline/ppo1/observations_adapter.py**
   - Observation processing
   - **Functions:** `build_high_level_observation()`, `build_horizon_observation()`

9. **kevin_integration/new_observations_pipeline/ppo1/rewards_adapter.py**
   - Reward computation
   - **Functions:** `compute_horizon_reward()`, `compute_tracking_error()`

10. **kevin_integration/new_observations_pipeline/ppo1/mulag_upper_level_env_cfg.py**
    - PPO hyperparameters
    - **Class:** `MulagUpperLevelPPORunnerCfg`

### 🧪 Testing
11. **kevin_integration/new_observations_pipeline/scripts/test_ppo1_idfd_smoke.py**
    - Smoke test for PPO1
    - Verifies ID/FD integration, task-space mode
    - Command: `./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/test_ppo1_idfd_smoke.py --num_envs 4`

### 💾 Model Checkpoints
12. **outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt**
    - Kevin's ID model (fine-tuned on IsaacLab PhysX)
    - Size: 111KB
    - Trained: Simscape → Fine-tuned: PhysX

13. **outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/fd_physx_finetuned_best.pt**
    - Kevin's FD model (fine-tuned on IsaacLab PhysX)
    - Size: 618KB
    - Trained: Simscape → Fine-tuned: PhysX

---

## ❌ Abandoned Pipeline (Don't Use!)

### ppo1_adapter/ Directory (All Files Failed)
- **kevin_integration/new_observations_pipeline/ppo1_adapter/task_to_cylinder_adapter.py** ❌
  - Learned adapter (failed to learn)
  - Outputs constants for all inputs
  - Zero correlation in training data

- **kevin_integration/new_observations_pipeline/ppo1_adapter/forward_response_model.py** ⚠️
  - Forward model (works but not needed)
  - Successfully trained but not part of working pipeline

- **kevin_integration/new_observations_pipeline/ppo1_adapter/analytical_adapter.py** ⚠️
  - Simple Jacobian approximation
  - Produces correct outputs but doesn't account for hydraulics

- **kevin_integration/new_observations_pipeline/scripts/ppo1_adapter/*** ❌
  - All training/testing scripts for failed adapter approach

**Why abandoned:**
- Training data had zero correlation (random single-step commands)
- Didn't use Kevin's hydraulic models
- Domain mismatch (IsaacLab vs Simscape)
- Fundamentally flawed approach

---

## 📁 Directory Structure

```
kevin_integration/
├── scripts/
│   └── train_upper_level_ppo_horizon_idfd.py ← ENTRY POINT
│
├── new_observations_pipeline/
│   ├── ppo1/ ← ✅ WORKING PIPELINE
│   │   ├── horizon_idfd_env_wrapper.py ← MAIN WRAPPER
│   │   ├── mulag_upper_level_env.py ← ISAACLAB ENV
│   │   ├── mulag_upper_level_env_cfg.py ← CONFIG
│   │   ├── observations_adapter.py
│   │   └── rewards_adapter.py
│   │
│   ├── low_level_controller/
│   │   ├── high_level_ee_controller.py ← JACOBIAN CONTROLLER
│   │   ├── idfd_rollout.py ← CLOSED-LOOP ROLLOUT
│   │   └── ...
│   │
│   ├── ppo1_adapter/ ← ❌ ABANDONED
│   │   └── (all files failed)
│   │
│   └── scripts/
│       └── test_ppo1_idfd_smoke.py ← SMOKE TEST
│
└── controllers/
    └── learned_models.py ← KEVIN'S ID/FD MODELS
```

---

## 🔑 Key Files Summary

| File | Role | Lines | Status |
|------|------|-------|--------|
| train_upper_level_ppo_horizon_idfd.py | Entry point | 200 | ✅ |
| horizon_idfd_env_wrapper.py | Main pipeline | 900 | ✅ |
| high_level_ee_controller.py | Jacobian | 200 | ✅ |
| idfd_rollout.py | Closed-loop | 50 | ✅ |
| learned_models.py | ID/FD models | 800 | ✅ |
| mulag_upper_level_env.py | IsaacLab env | 3000 | ✅ |
| mulag_upper_level_env_cfg.py | Config | 450 | ✅ |

---

## 🚀 Quick Start

```bash
# Test the pipeline
./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/test_ppo1_idfd_smoke.py --num_envs 4 --headless

# Train PPO1
./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \
  --num_envs 32 \
  --max_iterations 50 \
  --ee_action_mode task_space_delta \
  --headless
```

---

## 📊 Data Flow Summary

```
train_upper_level_ppo_horizon_idfd.py
    ↓ creates
HorizonIDFDEnv (horizon_idfd_env_wrapper.py)
    ↓ uses
HighLevelEEController (high_level_ee_controller.py)
    ↓ measures J_eff through
closed_loop_idfd_rollout (idfd_rollout.py)
    ↓ uses
Kevin's ID/FD Models (learned_models.py)
    ↓ executes in
MulagUpperLevelEnv (mulag_upper_level_env.py)
    ↓ simulates in
IsaacLab PhysX
```

---

## ✅ Files You Need

**Essential (cannot work without):**
1. train_upper_level_ppo_horizon_idfd.py
2. horizon_idfd_env_wrapper.py
3. high_level_ee_controller.py
4. idfd_rollout.py
5. learned_models.py
6. id_physx_finetuned_best.pt
7. fd_physx_finetuned_best.pt

**Important (environment):**
8. mulag_upper_level_env.py
9. mulag_upper_level_env_cfg.py
10. observations_adapter.py
11. rewards_adapter.py

**Optional (testing):**
12. test_ppo1_idfd_smoke.py

---

## 🗑️ Files You Can Ignore

Everything in:
- `kevin_integration/new_observations_pipeline/ppo1_adapter/`
- All `*adapter*.py` files in that directory
- All scripts in `scripts/ppo1_adapter/`

These were the failed approach and are no longer used.

---

## 📝 Configuration Checklist

Before training, verify:
- [ ] `ee_action_mode = "task_space_delta"` in mulag_upper_level_env_cfg.py
- [ ] ID checkpoint exists: id_physx_finetuned_best.pt
- [ ] FD checkpoint exists: fd_physx_finetuned_best.pt
- [ ] CUDA available: `nvidia-smi`
- [ ] IsaacLab environment activated: `conda activate env_isaaclab`

---

## 🎯 What Makes This Work

1. **Real Hydraulics:** Kevin's ID/FD models from Simscape
2. **Domain Adaptation:** Models fine-tuned on IsaacLab PhysX
3. **Empirical Jacobian:** Measured through full deployment chain
4. **Hierarchical Control:** PPO (task) → Jacobian (kinematics) → ID/FD (hydraulics)
5. **Closed-Loop Prediction:** Multi-step rollout with feedback

---

**The key insight:** Don't try to learn the Jacobian from bad data. Measure it empirically through the real system!
