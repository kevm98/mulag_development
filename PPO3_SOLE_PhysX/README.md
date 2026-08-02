# PPO3 — ROM-A terrain policy

PPO3 and PPO4 use the same verge task, sensors, 66-value observation, five-output
policy, rewards, safety logic, terrain, and evaluation gates. Their actuator is
the controlled comparison:

```text
PPO3: policy -> ROM-A -> PhysX
PPO4: policy -> ID -> FD -> PhysX
```

Run commands from `/home/rrlab/rrlab_mulag` with CUDA device `cuda:0`.

## Checkpoint

The shared-contract ROM checkpoint is:

```text
logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt
```

It is a behavior-preserving migration of the validated 54×4 ROM policy to the
shared 66×5 interface. Training must reset its optimizer.

## Commands

```bash
# Validate checkpoint structure
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO3_SOLE_PhysX/scripts/check_ready.py

# Visualize
./PPO3_SOLE_PhysX/visualize.sh

# Train/resume until held-out gates pass
./PPO3_SOLE_PhysX/train.sh

# Evaluate headlessly
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CKPT=logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt
$PY PPO3_SOLE_PhysX/scripts/evaluate.py "$CKPT" \
  --actuator rom --headless --device cuda:0 \
  --num_envs 32 --seed 4242 --steps 960 --stage 5 --eval_terrain \
  --drive_substep_interpolation true --report /tmp/ppo3_heldout.json

# Migrate another legacy 54×4 ROM checkpoint
$PY PPO3_SOLE_PhysX/scripts/migrate_shared_policy.py \
  /path/to/legacy.pt /path/to/shared_66x5.pt

# Tests
$PY -m pytest -q PPO3_SOLE_PhysX/tests PPO4_ID_FD_PhysX/tests
```

## Layout

```text
env.py, terrain_env.py          runtime and sensors
env_cfg.py, terrain_env_cfg.py shared task configuration
actions.py                     action filters and safety controllers
terrain_observations.py        shared 66-value observation
terrain_rewards.py             reward terms
terrain_convergence.py         held-out gates
scripts/train.py               one PPO process
scripts/train_until_converged.py training/evaluation supervisor
scripts/evaluate.py            headless evaluation and GUI
scripts/migrate_shared_policy.py checkpoint migration
tests/                         fast contracts
```
