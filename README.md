# RRLAB Mulag Hydraulic RL

This repository contains the Kevin/Mulag IsaacLab task-space reinforcement-learning workflow. The current objective is one fixed reachable `Messerkopf` end-effector pose per episode, accurate final position and orientation, low end-effector linear/angular speed, and 50 consecutive settled 10 ms control steps before strict success.

## Control Pipeline

```text
history/state -> frozen AAM -> z_arm_hat[24]
target/state/z_arm_hat -> PPO Action[4]
PPO -> qddot_cmd[5], qddot_cmd[4] = 0
qddot/state/z_arm_hat -> frozen ID -> ValveCmd[4]
ValveCmd/state -> frozen FD observer
ValveCmd/cylinder state/mechanical state -> frozen hydraulic ROM
ROM -> four cylinder position targets
measured EE pose/speed -> reward, telemetry, strict success
```

The observation remains 73D. PPO action remains 4D. AAM, ID, FD, and ROM checkpoints remain frozen. Mechanical 5D state and cylinder 4D writer state are separated explicitly; there is no runtime IK, MPC, QP, trajectory planner, or black-box replacement controller.

## Commands

Train from a fresh policy after reward or curriculum contract changes:

```bash
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 16 \
  --max_iterations 30 \
  --headless
```

Visualize an explicit checkpoint:

```bash
./rrlab.sh -p kevin_integration/scripts/play_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 1 \
  --checkpoint /absolute/path/to/model_250.pt
```

Evaluate checkpoints deterministically and write `checkpoint_evaluation.json`, `checkpoint_evaluation.csv`, and `best_checkpoint.json`:

```bash
./rrlab.sh -p kevin_integration/scripts/evaluate_mulag_checkpoints.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --run_dir /absolute/path/to/run \
  --num_envs 16 \
  --episodes 160 \
  --checkpoint_stride 10 \
  --headless
```

Play the documented best checkpoint from a run:

```bash
./rrlab.sh -p kevin_integration/scripts/play_mulag_adaptive_rl.py \
  --task Kevin-Mulag-HydraulicROM-TaskSpace-v0 \
  --num_envs 1 \
  --best_from_run /absolute/path/to/run
```

TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl
```

## Runtime Gates

Start with registration and one-environment visualization, then deterministic checkpoint evaluation, then repaired-reward training at 16 environments. Scale to 800 and 2000 environments only after rolling success rate, near-pose low-speed fraction, maximum hold duration, and position/orientation error trends improve without false velocity termination, target jumps, mapping mismatches, or nonfinite learned states.

## Reward Contract

The repaired contract is `task_space_reward_balance_v2`. It keeps strict success unchanged, adds potential-based progress, bounded avoidable actuator penalties, near-pose speed penalties, settling reward, and hold-progress reward. Reward-balance telemetry is logged under `reward_balance/*` and is not part of the observation.

Old checkpoints may be visualized only when tensor dimensions are compatible. Do not resume training silently across reward-contract or environment-contract changes; start a fresh PPO experiment unless a migration is explicitly documented.

Known limitation: static checks do not prove Isaac runtime success. The GUI/evaluation/training gates must still be run externally.
