# Trajectory-v2 PPO

This is a separate arm-workstation task. It uses a standalone arc-length 2-D
library with `line`, `sine_chain`, `circle`, and `rounded_square`. The existing
line-trained checkpoints and the legacy 42-D policy task are unchanged.

## Teacher preview and checks

```bash
PY=~/miniconda3/envs/env_isaaclab/bin/python

$PY verge_task/scripts/trajectory_2d_teacher.py \
  --output reports/trajectory_2d/teacher.svg \
  --stations 4 --station_spacing_m 2.0

$PY verge_task/scripts/trajectory_2d_preflight.py \
  --out reports/trajectory_2d/preflight.json
```

The teacher command is visualization-only. It also writes a route preview. The
approved v2 references use the measured head-down ground layer (`z=1.528 m` in
the truck frame), with line/sine at slow creep and circle/rounded-square at
parked stations. The preflight reports workspace reachability, command-cylinder
saturation, curvature/overshoot proxy, and head alignment. A passing preflight
is not a substitute for a PhysX rollout.

The physical scene keeps the original task presentation: a visual-only asphalt
road spans the transport route and a wide green verge spans the mower side. The
road/verge meshes do not change the validated collision ground. Closed loops
follow `park -> track one loop -> stow -> drive to the next station -> park`;
open line/sine paths use fixed low-speed chassis creep (`0.035`/`0.030 m/s`).

## Balanced PPO training

Start a new GPU policy because the geometry channels widen the observation from
42 to 45. Do not point this command at the historical line checkpoint.

```bash
$PY mulag_plant/policy_4cyl/scripts/train.py \
  --trajectory_2d --fresh_start --joint_finetune \
  --trajectory_shapes line,sine_chain,circle,rounded_square \
  --num_envs 4 --device cuda:0 --path_speed_mps 0.13 \
  --balanced_shape_rounds 3 --iterations_per_shape 4 \
  --save_interval 10
```

The phases alternate forward and reverse shape order, giving each shape the
same number of PPO iterations. Outputs go to
`logs/rsl_rl/ppo2_a_trajectory_2d/`; the old line logs are not overwritten.
The validated rollout uses four CUDA environments: larger replicated batches
can violate the PhysX clearance preflight. The 2-D task uses frame-consistent
clearance plus a 600-step station preposition and 180-step settle. Moving open
paths (`line`, `sine_chain`) use the frozen DLS fallback; PPO residuals remain
enabled for parked `circle` and `rounded_square` stations.

## Full-shape quality evaluation

Run the final checkpoint through one complete endpoint-to-endpoint traversal
of every approved shape:

```bash
$PY mulag_plant/policy_4cyl/scripts/evaluate_trajectory_2d.py \
  --checkpoint logs/rsl_rl/ppo2_a_trajectory_2d/<run>/model_<N>.pt \
  --shapes line,sine_chain,circle,rounded_square \
  --one_way --path_speed_mps 0.13 --device cuda:0 \
  --out reports/trajectory_2d/evaluation_final.json
```

The report includes tool and nearest-path error, normal error, peak error and
settling band, acceleration, residual/saturation fraction, head alignment,
clearance (including p95 and absolute transient), and truck drift. Acceptance
requires completion, p95 tool/normal/clearance error limits, head alignment,
and no truck drift; absolute clearance transients remain visible in the JSON.

## Visualization of a trained v2 checkpoint

```bash
$PY mulag_plant/policy_4cyl/scripts/play.py \
  --trajectory_2d --trajectory_shape rounded_square \
  --checkpoint logs/rsl_rl/ppo2_a_trajectory_2d/<run>/model_<N>.pt \
  --path_speed_mps 0.13 --device cuda:0
```

For the complete stop/drive/stow workflow and four repeated stations:

```bash
$PY mulag_plant/policy/scripts/stop_and_work.py \
  --trajectory_2d --checkpoint logs/rsl_rl/ppo2_a_trajectory_2d/<run>/model_<N>.pt \
  --station_shapes line,sine_chain,circle,rounded_square \
  --stations 4 --station_spacing_m 2.0 --drive_speed_mps 0.6 \
  --device cuda:0
```

Use `--zero_policy` with the same command to inspect the deterministic
head-down teacher and the road/verge scene before loading a PPO checkpoint.

The v2 observation appends nearest-point delta XY and signed normal error. The
reward adds a bounded normal-tracking term; the old task leaves both disabled.
