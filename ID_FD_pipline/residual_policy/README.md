# Residual PPO Policy for MULAG Hydraulic Mower Arm

Adds small residual corrections (cylinders 1-3 only) to the deterministic
head-down A->B->A arm controller. Cylinder 4 remains at the head-down teacher
value, and the truck remains completely stationary.

## Architecture

```
PPO Policy (3-dim residual)
    |
    v
Trajectory Generator -> Baseline Cylinder Goal (4-dim)
    |
    v
Residual Correction (cyls 1-3 only) -> Corrected Cylinder Goal
    |
    v
Cylinder DLS Controller -> ID/FD Pipeline -> PhysX Target Drives
```

The policy does NOT replace:
- The head-down trajectory generator
- Cylinder DLS
- EE cascade
- ID/FD hydraulic models
- The reference-dominant ID/FD blend
- Cylinder 4 head orientation control
- Existing safety limits

## Action Definition

- **Action dimension**: 3
- **action[0]**: residual correction to cylinder 1 target
- **action[1]**: residual correction to cylinder 2 target
- **action[2]**: residual correction to cylinder 3 target
- **Cylinder 4**: no policy action, stays at head-down teacher value

Residual application:
```
raw_action = clamp(policy_action, -1, 1)
filtered_action = 0.8 * prev_filtered + 0.2 * raw_action
corrected_goal[0:3] = baseline_goal[0:3] + residual_scale_m * filtered_action
corrected_goal[3] = baseline_goal[3]
```

## Observation Layout (41 dimensions)

| Slice  | Content                           |
|--------|-----------------------------------|
| 0:3    | tool_error_truck                  |
| 3:6    | measured_tool_velocity_truck      |
| 6:9    | reference_tool_velocity_truck     |
| 9:12   | reference_tool_acceleration_truck |
| 12:16  | measured_cylinder_extensions      |
| 16:20  | cylinder_goal_error               |
| 20:24  | measured_cylinder_velocities      |
| 24:27  | previous_filtered_policy_action   |
| 27:31  | current_valve_commands            |
| 31:35  | corrected_fd_delta                |
| 35:37  | trajectory_phase_sin_cos          |
| 37     | trajectory_direction (+1/-1)      |
| 38     | cutter_bottom_world_clearance     |
| 39     | head_down_alignment               |
| 40     | head_down_alignment_rate          |

## Reward

```
tracking  = 4.0 * exp(-(error_norm / 0.02)^2)
progress  = 0.5 * clamp((prev_error - error) / 0.005, -1, 1)
action_mag = -0.02 * mean(filtered_action^2)
action_rate = -0.05 * mean((filtered - prev_filtered)^2)
head_pen   = -2.0 * relu((0.995 - head_alignment) / 0.02)^2
ground_pen = -2.0 * relu((abs(clearance) - 0.10) / 0.02)^2
sat_pen    = -1.0 * mean(abs(valve) >= 0.99)

total = tracking + progress + action_mag + action_rate
      + head_pen + ground_pen + sat_pen
```

Reward is always computed against the **original unmodified tool reference**.

## Training

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/residual_policy/scripts/train.py \
  --headless --num_envs 64 --max_iterations 1000 \
  --curriculum_stage 1 --seed 17
```

Curriculum stages:
- Stage 1: small deterministic residual (+-1mm, 0.10 m/s)
- Stage 2: nominal speed (+-2mm, 0.13 m/s)
- Stage 3: expanded residual authority (+-4mm, 0.13 m/s)

Each stage is an explicit training run. Resume the previous checkpoint with
`--checkpoint`; the script does not silently advance stages.

On the 6 GB RTX 3060, 64 environments is the measured throughput sweet spot:
383 simulation steps/s versus 276 at 32. With 128 environments the physical
reset did not complete an iteration within three minutes, so larger batches
are slower for this articulated plant. A 64-env × 1,000-iteration run collects
the same 8.192 million transitions as 32 envs × 2,000 iterations, with fewer
PPO updates; evaluate the checkpoint and extend training if it has not converged.

## Evaluation

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/residual_policy/scripts/evaluate.py \
  --headless --checkpoint /absolute/path/to/model.pt --num_seeds 5
```

## Zero-Action Contract

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/residual_policy/tests/zero_action_contract.py \
  --headless --num_envs 2
```

The reset contract is physical: the arm uses a 300-step cosine deployment to
the closest verified endpoint and a 120-step hold. Policy actions, reward,
termination checks, and the 12-second training horizon begin only after this
phase. At handoff, the ID/FD and DLS histories are reset so deployment state
cannot leak into the scored episode.

Training resets sample a position and direction over the complete A→B→A path,
cosine-deploy to that verified teacher pose, and accelerate from rest toward
the next endpoint. This lets 12-second episodes cover the entire 4.067 m arc
without an artificial velocity jump. Play and evaluation disable randomized
starts and run a full cycle from the closest endpoint.

Validated zero-action full-cycle result (seed 17, 3,874 scored steps):

- corrected goal vs deterministic baseline: exactly 0.0 m
- cylinder-4 policy delta: exactly 0.0 m
- mean / p95 / max tool error: 0.01040 / 0.02324 / 0.03093 m
- maximum truck planar drift: 0.00151 m

Validated 32-environment stage-1 preflight after batch-controller correction:

- maximum 100-step cutter clearance: 0.11883 m (hard limit 0.15 m)
- maximum 100-step tool error: 0.03315 m
- one PPO iteration: 4,096 samples, zero safety terminations
- iteration mean / p95 tracking error: 0.0107 / 0.0248 m

## Truck: Completely Stationary

- Scripted vehicle motion is disabled
- Wheel positions are held at initial values
- Planar chassis drift is corrected (1 mm deadband)
- Suspension Z motion is not constrained
- Episodes terminate if truck XY drift exceeds 0.02 m

## Cylinder 4: Deterministic Only

- Cylinder 4 always stays at the head-down teacher value
- No policy action dimension for cylinder 4
- The EE cascade Jacobian has column 4 zeroed out

## Configuration

See `ResidualPolicyEnvCfg` in `env_cfg.py` for all configurable parameters.

Key defaults:
- residual_scale_m: 0.002 (+-2 mm)
- action_filter: 0.8
- path_speed_mps: 0.13
- episode_length_s: 12.0 of scored motion (deployment is excluded)
