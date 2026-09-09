# ID/FD Fine-Tuning Infrastructure

This directory holds fine-tuned ID and FD checkpoints if original models
are proven insufficient for paper-matched servo performance.

## Rules

- Never overwrite original checkpoints in `Models_cylinder /`
- Dataset must contain dynamic data at actual control rate
- Include positive and negative velocity commands
- Include one-cylinder and multi-cylinder commands
- Include zero-command drift, low/mid/high extension regions

## When to Fine-Tune

- **ID**: Stage A failure (valve_ff near zero for nonzero desired velocity)
- **FD**: Stage C failure (valve_total nonzero but fd_delta tiny)

## ID Fine-Tuning

Input: desired velocity[4], extension[4], extension_vel[4], joint_pos[10], joint_vel[10]
Output: valve command that achieved desired velocity[4]
Labels: from successful PID servo rollouts or inverse search

## FD Fine-Tuning

Input: valve_total[4], extension[4], extension_vel[4], joint_pos[10], joint_vel[10]
Output: measured DeltaExtension_next[4]

## Post Fine-Tuning

1. Rerun `tune_cylinder_velocity_servo.py` with new checkpoints
2. Rerun PhysX audit
3. Rerun upper-level tracking
