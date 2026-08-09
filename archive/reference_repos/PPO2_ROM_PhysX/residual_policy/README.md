# PPO2 ROM residual policy

This is the ROM-only, trajectory-level counterpart of the PPO1-ID-FD residual policy. It keeps
the policy and task contract aligned 1:1 with PPO1 while replacing the complete learned ID/FD
actuator with one command-conditioned hydraulic ROM.

```text
trajectory library -> baseline 4-cylinder goal
                         |
41-D observation -> PPO2 3-D residual (cylinders 1-3 only)
                         |
                 corrected cylinder goal
                         |
                       DLS
                         |
       command-conditioned ROM -> bounded PhysX target
                         |
       measured tool/cylinder/hydraulic feedback
```

The policy does not command cylinder 4, joint positions, valves, pressure, force, or truck
motion directly. Cylinder 4 remains owned by the deterministic head-down teacher. The truck is
position-locked while the arm is tracking.

## PPO1/PPO2 parity contract

| Contract | PPO1 ID-FD residual | PPO2 ROM residual |
|---|---|---|
| Observation | 41-D | same 41-D |
| Policy action | 3-D residual | same 3-D residual |
| Actor/critic | `[256,128,64]`, ELU | same |
| Reward and safety terms | tool tracking, progress, action, head, clearance, saturation | same |
| Shapes | line, arc, circle, sine | same |
| Curriculum | stages 1-4 | same |
| Episode | 12 s scored after physical deploy/settle | same |
| Truck during arm work | stationary | stationary |
| DLS settings | fast direct-target branch with velocity feedforward | same |
| Hydraulic backend | ID -> FD | command ROM only |
| Log keys | residual/tool/safety metrics | same |
| Log root | `logs/rsl_rl/id_fd_residual/` | `logs/rsl_rl/ppo2_rom_residual/` |

The command ROM is reset at every episode handoff. Disabled ID and FD runtimes are neither loaded
nor reset.

## Observation and action

The observation layout is identical to PPO1:

| Slice | Signal |
|---|---|
| `0:3` | truck-frame tool error |
| `3:6` | measured tool velocity |
| `6:9` | reference tool velocity |
| `9:12` | reference tool acceleration |
| `12:16` | measured cylinder extensions |
| `16:20` | corrected cylinder-goal error |
| `20:24` | measured cylinder velocities |
| `24:27` | previous filtered policy action |
| `27:31` | ROM-predicted valve |
| `31:35` | ROM-predicted achievable extension increment |
| `35:37` | trajectory phase sine/cosine |
| `37` | A->B/B->A direction |
| `38` | absolute or reference-relative clearance |
| `39` | head-down alignment |
| `40` | head-down alignment rate |

The policy action is applied as

```text
raw = clamp(policy_action, -1, 1)
filtered = 0.8 * previous_filtered + 0.2 * raw
corrected_goal[0:3] = clamp(baseline_goal[0:3] + residual_scale * filtered)
corrected_goal[3] = baseline_goal[3]
```

See [`../../Universal_Policy/README.md`](../../Universal_Policy/README.md) for the complete PPO,
reward, GAE, observation, and termination equations.

## Do not confuse it with the compact PPO2 tasks

`PPO2_ROM_PhysX/scripts/train_ppo2.py --task cylinder` trains a fast 25-observation, 4-action
random-cylinder-setpoint experiment. `--task ee` trains a separate 30-observation, 4-action
development task. Neither is the multi-shape residual policy documented here, and their
checkpoints are incompatible with this 41-observation, 3-action policy.

The dedicated entry point is:

```text
PPO2_ROM_PhysX/residual_policy/scripts/train.py
```

## Required preflight

Run from `/home/rrlab/rrlab_mulag` before starting a long run:

```bash
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

# Four-axis ROM authority and return behavior
${PY} PPO2_ROM_PhysX/dls_solver/tests/rom_step_gate.py \
  --headless --goal_delta_m 0.035 --steps 600 --tolerance_m 0.01 \
  --backend rom --dynamics_model command_rom --hydraulic_mode fd_dp_state

# Fast/smooth deterministic ROM trajectory
${PY} PPO2_ROM_PhysX/dls_solver/tests/ground_arc_tracking_gate.py \
  --headless --speed_mps 0.18 --cycles 1 \
  --out outputs/ppo2_rom_physx/ground_arc_validation.json

# ROM residual wrapper must reproduce its zero-action baseline exactly
${PY} PPO2_ROM_PhysX/residual_policy/tests/zero_action_contract.py \
  --headless --num_envs 2

# Every stage-4 shape with zero residual
${PY} PPO2_ROM_PhysX/residual_policy/scripts/validate_workspace_shapes.py \
  --headless --speed_mps 0.16 \
  --out outputs/ppo2_rom_residual/workspace_shape_baseline.json
```

Do not begin PPO if ROM reset, zero-action equality, cylinder 4 isolation, truck stationarity,
or the deterministic shape baseline fails.

## Long staged training

Each stage below uses 64 environments, 128 steps per environment, and 2,000 PPO iterations:
16.384 million samples per stage. Each command creates a new timestamped directory below
`logs/rsl_rl/ppo2_rom_residual/`. Select and evaluate a checkpoint before using it to initialize
the next stage.

```bash
cd /home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

# Stage 1: legacy arc, 0.10 m/s, ±1 mm; train from scratch
${PY} PPO2_ROM_PhysX/residual_policy/scripts/train.py \
  --headless --num_envs 64 --max_iterations 2000 --seed 17 \
  --curriculum_stage 1

# Replace this with the selected stage-1 checkpoint
STAGE1=/absolute/path/to/logs/rsl_rl/ppo2_rom_residual/stage1/model_N.pt

# Stage 2: legacy arc, 0.13 m/s, ±2 mm
${PY} PPO2_ROM_PhysX/residual_policy/scripts/train.py \
  --headless --num_envs 64 --max_iterations 2000 --seed 17 \
  --curriculum_stage 2 --checkpoint "${STAGE1}"

STAGE2=/absolute/path/to/logs/rsl_rl/ppo2_rom_residual/stage2/model_N.pt

# Stage 3: legacy arc, 0.13 m/s, ±4 mm
${PY} PPO2_ROM_PhysX/residual_policy/scripts/train.py \
  --headless --num_envs 64 --max_iterations 2000 --seed 17 \
  --curriculum_stage 3 --checkpoint "${STAGE2}"

STAGE3=/absolute/path/to/logs/rsl_rl/ppo2_rom_residual/stage3/model_N.pt

# Stage 4: line/arc/circle/sine, 0.16 m/s, ±10 mm
${PY} PPO2_ROM_PhysX/residual_policy/scripts/train.py \
  --headless --num_envs 64 --max_iterations 2000 --seed 17 \
  --curriculum_stage 4 --trajectory_shapes line,arc,circle,sine \
  --checkpoint "${STAGE3}"
```


Although the tensor shapes match, the hydraulic feedback distribution is different.

## Logs

The ROM residual run writes the same metric names as PPO1, including:

```text
reward/total
reward/tracking
reward/progress
reward/action_magnitude
reward/action_rate
reward/head_penalty
reward/ground_penalty
reward/saturation_penalty
error/mean
error/p95
error/max
head/alignment_min
truck/drift_max
residual/mean_abs
residual/max_abs
termination/*
```

Launch TensorBoard with:

```bash
tensorboard --logdir /home/rrlab/rrlab_mulag/logs/rsl_rl/ppo2_rom_residual
```

## Evaluation

```bash
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CANDIDATE=/absolute/path/to/logs/rsl_rl/ppo2_rom_residual/run/model_N.pt

${PY} PPO2_ROM_PhysX/residual_policy/scripts/evaluate.py \
  --headless --checkpoint "${CANDIDATE}" \
  --curriculum_stage 4 --num_seeds 5 --seed 17 \
  --out outputs/ppo2_rom_residual/stage4_evaluation.json

${PY} PPO2_ROM_PhysX/residual_policy/scripts/play.py \
  --checkpoint "${CANDIDATE}" --curriculum_stage 4 \
  --trajectory_shape arc --realtime
```

With no `--trajectory_shape`, evaluation compares policy and zero residual on line, arc, circle,
and sine for every seed. Promote only a checkpoint that finishes all cycles, improves mean and
p95 tool error, preserves clearance/head/truck limits, and passes oscillation validation.

## Pure contract checks

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  PPO2_ROM_PhysX/residual_policy/tests/observation_contract.py \
  PPO2_ROM_PhysX/residual_policy/tests/action_bounds.py \
  PPO2_ROM_PhysX/residual_policy/tests/reward_anti_gaming.py \
  PPO2_ROM_PhysX/residual_policy/tests/cylinder4_isolation.py \
  PPO2_ROM_PhysX/tests/test_command_rom_runtime.py
```
