# PPO4 — ID/FD terrain policy

PPO4 shares PPO3's terrain, sensors, 66-value observation, five policy outputs,
controller, rewards, safety logic, and evaluation gates. Only the actuator path
differs:

```text
PPO3: policy -> ROM-A -> PhysX
PPO4: policy -> ID -> FD -> PhysX
```

Run commands from `/home/rrlab/rrlab_mulag` on `cuda:0`.

## Experiments

| experiment | contract | what it is |
|---|---|---|
| `ppo4_sole_idfd/shared_v2_idfd` | 66×5 | previous ID/FD policy, `model_validated.pt` |
| `ppo4_sole_idfd_contact_safe_v1` | **96×5** | the contact-safe policy |

## Contact safety

PPO4 inherits the whole contact-safety task from PPO3 — the state machine, the
seven reward terms, the six observation channels, the strict gates. Nothing
about it is defined here, and the contract test forbids PPO4 from assigning
any of those fields: if PPO4 set its own gap tolerance or acquisition
hysteresis, "PPO4 lifts more than PPO3" would stop being a statement about the
actuator and become a statement about two different definitions of lifting.

Measured pre-fix on the held-out verge (stage 5, s_curve, relief 1.0, 960
steps × 32 envs), PPO4 is the worse of the two:

| | PPO3 / ROM-A | PPO4 / ID-FD |
|---|---:|---:|
| ground contact (gate ≥ 0.90) | 0.9486 PASS | 0.8945 FAIL |
| visible lifts (> 5 mm) | 334 | 359 |
| max consecutive airborne steps | 45 | 79 |
| max airborne gap | 0.297 m | 0.675 m |
| max upward cutter velocity | 4.41 m/s | 7.93 m/s |
| max contact force | 119 kN | 170 kN |
| max cylinder target lead | 0.0300 m | 0.0406 m |

The mechanism is shared, and so is the field that causes it:
`cylinder_target_max_lead_m` is read unconditionally in
`mulag_upper_level_env._run_cylinder_id_pipeline` on the path BOTH actuators
take, and both run saturated against its inherited 25 mm. See
[`../PPO3_SOLE_PhysX/reports/contact_safe_v1/PREFIX_DIAGNOSTIC.md`](../PPO3_SOLE_PhysX/reports/contact_safe_v1/PREFIX_DIAGNOSTIC.md).

## Commands

```bash
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

# Migrate the 66x5 checkpoint to the 96x5 contact-safe contract
$PY PPO3_SOLE_PhysX/scripts/migrate_shared_policy.py \
  logs/rsl_rl/ppo4_sole_idfd/shared_v2_idfd/model_validated.pt \
  logs/rsl_rl/ppo4_sole_idfd_contact_safe_v1/model_seed_96.pt \
  --mode widen_observation --target_observations 96

# Visualize
./PPO4_ID_FD_PhysX/visualize.sh

# Train the contact-safe policy (PPO3's scripts, --actuator idfd)
bash PPO3_SOLE_PhysX/scripts/run_contact_safe_training.sh idfd

# Evaluate the contact-safe policy with the strict gates
CKPT=logs/rsl_rl/ppo4_sole_idfd_contact_safe_v1/model_best.pt
$PY PPO3_SOLE_PhysX/scripts/evaluate.py "$CKPT" \
  --actuator idfd --headless --device cuda:0 \
  --num_envs 32 --seed 17 --steps 960 --stage 5 --eval_terrain \
  --contact_observation true --contact_safety_reward true \
  --model_error_history 3 --strict_contact \
  --drive_substep_interpolation true --report /tmp/ppo4_heldout.json

# Measure contact loss event-by-event
$PY PPO3_SOLE_PhysX/scripts/probe_contact_loss.py \
  --checkpoint "$CKPT" --actuator idfd --headless --device cuda:0 \
  --num_envs 32 --seed 17 --steps 960 --stage 5 --eval_terrain \
  --path_family s_curve --terrain_height_scale 1.0 \
  --contact_observation true --model_error_history 3 \
  --report /tmp/ppo4_contact_loss.json

# Evaluate the previous 66x5 checkpoint (unchanged)
$PY PPO3_SOLE_PhysX/scripts/evaluate.py \
  logs/rsl_rl/ppo4_sole_idfd/shared_v2_idfd/model_validated.pt \
  --actuator idfd --headless --device cuda:0 \
  --num_envs 32 --seed 4242 --steps 960 --stage 5 --eval_terrain \
  --drive_substep_interpolation true --report /tmp/ppo4_heldout_66.json

# Check the viewer/runtime path
$PY PPO4_ID_FD_PhysX/scripts/check_ready.py \
  logs/rsl_rl/ppo4_sole_idfd/shared_v2_idfd/model_validated.pt \
  --headless --device cuda:0

# Contract tests
$PY -m pytest -q PPO3_SOLE_PhysX/tests PPO4_ID_FD_PhysX/tests
```

PPO4 defines only ID/FD model/runtime configuration; shared task changes belong
in `PPO3_SOLE_PhysX` so both policies remain directly comparable.
