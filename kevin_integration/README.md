# Kevin Mulag Fixed Final-Pose Control

This directory contains the canonical Kevin Mulag workflow:

```text
fixed robot-base EE target
-> 73D task-space observation
-> PPO Action[4]
-> qddot_cmd[5], qddot_cmd[:, 4] = 0
-> frozen ID conditioned by z_arm_hat[24]
-> ValveCmd[4]
-> frozen FD observer + valve-conditioned hydraulic ROM
-> four cylinder position targets
-> IsaacLab Mulag mechanism
```

![Current control architecture](Control_Pipeline.png)

The active Gym task is:

```text
Kevin-Mulag-HydraulicROM-TaskSpace-v0
```

The final-pose target is sampled once for each selected environment at reset,
uses `[x, y, z, qw, qx, qy, qz]` in the `robot_base` frame, and remains fixed
for the episode. PPO is the task-space-to-action bridge; there is no runtime IK.

## Environment

Run commands from the repository root:

```bash
cd /home/rrlab/rrlab_mulag
conda activate env_isaaclab
```

## Run The Simulation

### 1. Visible scripted pipeline validation

This opens IsaacLab with one environment and exercises the learned/hydraulic
pipeline without claiming PPO task-space tracking:

```bash
./rrlab.sh -p kevin_integration/scripts/play_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 1 \
  --play_mode joint_space_validation
```

### 2. Fixed final-pose PPO playback

Use an existing PPO checkpoint:

```bash
CHECKPOINT=/absolute/path/to/model_250.pt

./rrlab.sh -p kevin_integration/scripts/play_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 1 \
  --checkpoint "$CHECKPOINT" \
  --play_mode rl_task_space_tracking
```

Add `--headless` for non-visual playback.

On the current development machine, an example local checkpoint is
`kevin_integration/results/stage1_500env_13p9M_run/model_250.pt`. PPO `.pt`
artifacts are ignored by Git and are therefore not published with the source.

### 3. Runtime preflight only

This starts IsaacLab, runs the built-in asset and learned-pipeline motion
preflights, and exits before creating the PPO runner:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 1 \
  --max_iterations 0 \
  --preflight_only \
  --headless
```

### 4. PPO training

Small validation run:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 16 \
  --max_iterations 30 \
  --headless
```

Larger run after the one- and sixteen-environment gates pass:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 600 \
  --max_iterations 100 \
  --headless
```

Use 2,000 environments only after the smaller runs are stable:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 2000 \
  --max_iterations 100 \
  --headless
```

Resume training from a checkpoint:

```bash
CHECKPOINT=/absolute/path/to/model_250.pt

./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 600 \
  --max_iterations 100 \
  --checkpoint "$CHECKPOINT" \
  --headless
```

## Online AAM Branch

The Arm Adaptation Module is active online and remains frozen:

```text
q[5], qdot[5], dP[4], previous applied ValveCmd[4]
-> one AAM history with shape (N, 100, 18)
-> frozen aam.pth under inference mode
-> z_arm_hat[24]
   |-> PPO observation indices 44:68
   `-> matching ID conditioning input
```

`_prev_valve_cmd` is copied from the ValveCmd supplied to FD and ROM during the
preceding control step. The same stored latent is used by PPO and ID. AAM does
not directly write valves, torques, ROM state, cylinder targets, rewards,
targets, or success state. The System Factor Encoder is offline-only.

## 73D Observation

```text
0:7    final target EE pose
7:14   current EE pose
14:17  position error
17:18  quaternion-geodesic orientation error
18:23  mechanical q[5]
23:28  mechanical qdot[5]
28:32  cylinder position[4]
32:36  cylinder velocity[4]
36:40  previous Action[4]
40:44  previous applied ValveCmd[4]
44:68  z_arm_hat[24]
68:72  normalized FD observer output[4]
72:73  vehicle speed[1]
```

## Active Hydraulic ROM

Only the deployed valve-conditioned causal ROM is command-authoritative:

```text
kevin_integration/rom_cylinders/exported_valve_conditioned_causal_rom/
```

Runtime dependencies:

- `valve_conditioned_causal_rom.py`
- `multirate_rom_wrapper.py`
- `cylinder_coordinate_adapter.py`
- `cylinder_coordinate_mapping.json`
- `rom_data_contract.py`

The ROM consumes the fixed 26-feature order:

```text
ValveCmd[4], dP[4], Extension[4], ExtensionVelocity[4],
JointPos[5], JointVel[5]
```

It predicts `DeltaExtension[4]` over `0.05 s`. The multirate wrapper distributes
that prediction over five `0.01 s` Isaac control steps. The coordinate adapter
maps Simscape extension deltas to four Isaac prismatic position targets. FD is
observer-only and does not write actuator effort in this workflow.

## Current Source Layout

```text
scripts/
  train_mulag_adaptive_rl.py
  play_mulag_adaptive_rl.py

tasks/
  registered environment, configuration, task/reference helpers, contract tests

rl/
  AAM/ID/FD controller, observation/action adapters, rewards, torque conversion

controllers/
  learned checkpoint loaders and the imported reduced-ROM compatibility class

rom_cylinders/
  deployed valve-conditioned ROM, export package, mapping, multirate wrapper,
  data contract, and focused tests

models/
  aam.pth, id.pth, fd.pth, fd_scaler.pth, sd.pth

utils/
  Isaac simulation memory configuration
```

The reduced hydraulic actuator class remains because the environment imports
and initializes it for the retained compatibility path. It is not the active
command-authoritative ROM in the canonical task.

## Fixed Final-Pose Contract

The canonical experiment keeps:

- automatic curriculum advancement disabled;
- Stage 1 action authority `0.25`;
- episode duration `15 s`;
- position tolerance `0.05 m`;
- orientation tolerance `0.0872664626 rad`;
- EE linear speed tolerance `0.05 m/s`;
- EE angular speed tolerance `0.10 rad/s`;
- success hold of 50 consecutive `0.01 s` steps;
- separate success termination and one-time bonus `10.0`.

## Offline Verification

These commands do not start IsaacLab:

```bash
python kevin_integration/tasks/test_aam_runtime_wiring_contract.py
python kevin_integration/tasks/test_fixed_final_pose_task.py
python kevin_integration/tasks/test_curriculum_learnability_repair.py
python kevin_integration/tasks/test_preflight_observation_contract.py
python kevin_integration/tasks/test_asset_preflight_classifier.py
python kevin_integration/tasks/test_mechanical_constraint_resolver.py
python kevin_integration/tasks/test_rrlab_reference_conventions.py
python kevin_integration/tasks/test_task_space_reference.py
python kevin_integration/tasks/test_fd_observer_contract.py
python kevin_integration/rom_cylinders/test_valve_conditioned_runtime.py
python kevin_integration/rom_cylinders/test_multirate_rom_wrapper.py
python kevin_integration/rom_cylinders/test_cylinder_coordinate_adapter.py
```

Offline tests verify source and tensor contracts. They do not prove live Isaac
motion, GPU scaling, or training quality. Run the preflight and simulation
commands above before claiming runtime success.

## More Detail

- [Active file map](ACTIVE_PIPELINE_FILES.md)
- [IsaacLab control contract](ISAACLAB_CONTROL_CONTRACT.md)
- [Full hydraulic pipeline](FULL_HYDRAULIC_PIPELINE.md)
- [ROM runtime contract](rom_cylinders/ROM_DATA_CONTRACT.md)
