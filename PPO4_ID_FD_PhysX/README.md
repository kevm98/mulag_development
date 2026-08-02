# PPO4 — ID/FD terrain policy

PPO4 shares PPO3's terrain, sensors, 66-value observation, five policy outputs,
controller, rewards, safety logic, and evaluation gates. Only the actuator path
differs:

```text
PPO3: policy -> ROM-A -> PhysX
PPO4: policy -> ID -> FD -> PhysX
```

Run commands from `/home/rrlab/rrlab_mulag` on `cuda:0`.

## Checkpoint

```text
logs/rsl_rl/ppo4_sole_idfd/shared_v2_idfd/model_validated.pt
```

This checkpoint passes all 17 held-out gates with the shared head controller.

## Commands

```bash
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CKPT=logs/rsl_rl/ppo4_sole_idfd/shared_v2_idfd/model_validated.pt

# Visualize
./PPO4_ID_FD_PhysX/visualize.sh

# Train/resume
./PPO4_ID_FD_PhysX/train.sh

# Evaluate headlessly
$PY PPO3_SOLE_PhysX/scripts/evaluate.py "$CKPT" \
  --actuator idfd --headless --device cuda:0 \
  --num_envs 32 --seed 4242 --steps 960 --stage 5 --eval_terrain \
  --drive_substep_interpolation true --report /tmp/ppo4_heldout.json

# Check the viewer/runtime path
$PY PPO4_ID_FD_PhysX/scripts/check_ready.py \
  "$CKPT" --headless --device cuda:0

# Contract tests
$PY -m pytest -q PPO3_SOLE_PhysX/tests PPO4_ID_FD_PhysX/tests
```

PPO4 defines only ID/FD model/runtime configuration; shared task changes belong
in `PPO3_SOLE_PhysX` so both policies remain directly comparable.
