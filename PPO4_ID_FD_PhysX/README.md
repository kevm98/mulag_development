# PPO4 — sole terrain policy with ID/FD dynamics

PPO4 runs the PPO3 verge-mowing task with the validated PPO1 ID/FD actuator:

```text
policy -> ID (target to valve) -> FD (valve to cylinder motion) -> PhysX
```

The terrain, rewards, observations, safety gates, and training code are shared
with PPO3. PPO4 only supplies the ID/FD environment configuration.

## Requirements

- IsaacLab: `/home/rrlab/IsaacLab`
- Conda environment: `env_isaaclab`
- CUDA device: `cuda:0`
- ID model: `PPO1_ID_FD_PhysX/models/id/best_id_cylinder_control.pt`
- FD model: `PPO1_ID_FD_PhysX/models/fd/best_fd_cylinder_strict.pt`

Run commands from the repository root:

```bash
cd /home/rrlab/rrlab_mulag
```

## Train

The launcher resumes from the current best checkpoint by default. Pass another
compatible five-action ID/FD checkpoint as the first argument when needed.

```bash
./PPO4_ID_FD_PhysX/train.sh

# Custom checkpoint
./PPO4_ID_FD_PhysX/train.sh /path/to/model.pt
```

Training uses the shared PPO3 convergence supervisor, fixed q5-to-c4 gain
`0.4`, and held-out rollback checks.

## Evaluate

```bash
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CHECKPOINT=logs/rsl_rl/ppo4_sole_idfd/q5_hydraulic_gain040_stable_v6/convergence/model_best.pt

$PY PPO3_SOLE_PhysX/scripts/evaluate.py "$CHECKPOINT" \
  --actuator idfd --headless --device cuda:0 \
  --num_envs 64 --seed 17 --steps 960 --stage 5 --eval_terrain \
  --head_joint_action --head_action_to_c4_gain 0.4 \
  --drive_substep_interpolation true \
  --report /tmp/ppo4_heldout.json
```

## Visualize

```bash
./PPO4_ID_FD_PhysX/visualize.sh

# Custom checkpoint or additional evaluator arguments
./PPO4_ID_FD_PhysX/visualize.sh /path/to/model.pt --path_family lane_change_hold
```

## Readiness check

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO4_ID_FD_PhysX/scripts/check_ready.py \
  logs/rsl_rl/ppo4_sole_idfd/q5_hydraulic_gain040_stable_v6/convergence/model_best.pt \
  --headless --device cuda:0
```

## Tests

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  PPO3_SOLE_PhysX/tests PPO4_ID_FD_PhysX/tests -q
```

The current checkpoint passes 15 of 17 held-out gates. Tracking, contact,
smoothness, and acceleration pass; the remaining failures are the p01 and hard
minimum head-alignment gates. q5 is retained for checkpoint compatibility but
does not add an independent physical control direction.
