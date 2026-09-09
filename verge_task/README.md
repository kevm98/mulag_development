# PPO3 — ROM-A terrain policy

PPO3 and PPO4 use the same verge task, sensors, 66-value observation, five-output
policy, rewards, safety logic, terrain, and evaluation gates. Their actuator is
the controlled comparison:

```text
PPO3: policy -> ROM-A -> PhysX
PPO4: policy -> ID -> FD -> PhysX
```

Run commands from `~/rrlab_mulag` with CUDA device `cuda:0`.

## Experiments

| experiment | contract | what it is |
|---|---|---|
| `ppo3_sole_terrain/shared_v2_rom` | 66×5 | the previous production ROM policy, `model_migrated.pt` |
| `ppo3_sole_rom_contact_safe_v1` | **96×5** | the contact-safe policy (see below) |

The contact-safe experiment adds six contact-state observation channels and
raises the model-error history to depth 3, so its observation is 96 wide and
the 66×5 checkpoint is not loadable into it directly. Migrate, do not force:

```bash
PY=~/miniconda3/envs/env_isaaclab/bin/python
$PY verge_task/scripts/migrate_shared_policy.py \
  logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt \
  logs/rsl_rl/ppo3_sole_rom_contact_safe_v1/model_seed_96.pt \
  --mode widen_observation --target_observations 96
```

New columns get zero weight and a neutral normalizer, so the migrated network
computes exactly what the old one computed until training moves them. The
optimizer must be reset (`--reset_optimizer_on_resume`), and the migration
records that requirement in the checkpoint.

## Contact safety

`ground_contact_fraction` is a DUTY CYCLE and cannot see the cutter lifting:
the previous production checkpoint scores 0.9486 on it — a pass — while
spending up to 45 consecutive control steps and 297 mm off the verge. The
measurement, the mechanism and the traces are in
[`reports/contact_safe_v1/PREFIX_DIAGNOSTIC.md`](reports/contact_safe_v1/PREFIX_DIAGNOSTIC.md);
the full write-up is [`reports/CONTACT_SAFE_V1_REPORT.md`](reports/CONTACT_SAFE_V1_REPORT.md).

Nine strict promotion gates (zero visible lifts, zero impact ejections, zero
hard impacts, zero consecutive airborne steps after support acquisition) are
scored with `--strict_contact`. They are deliberately kept OUT of the
historical 17-gate set so every stored report stays comparable.

## Commands

```bash
PY=~/miniconda3/envs/env_isaaclab/bin/python

# Validate checkpoint structure
$PY verge_task/scripts/check_ready.py

# Visualize
./verge_task/visualize.sh

# Measure contact loss event-by-event (the pre-fix diagnostic)
$PY verge_task/scripts/probe_contact_loss.py \
  --checkpoint logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt \
  --actuator rom --headless --device cuda:0 \
  --num_envs 32 --seed 17 --steps 960 --stage 5 \
  --eval_terrain --path_family s_curve --terrain_height_scale 1.0 \
  --report /tmp/ppo3_contact_loss.json

# Impact-energy sweep (plant only, shipped policy held fixed)
bash verge_task/scripts/run_impact_energy_sweep.sh

# Train the contact-safe policy (three grace-curriculum stages)
bash verge_task/scripts/run_contact_safe_training.sh rom

# Evaluate the contact-safe policy with the strict gates
CKPT=logs/rsl_rl/ppo3_sole_rom_contact_safe_v1/model_best.pt
$PY verge_task/scripts/evaluate.py "$CKPT" \
  --actuator rom --headless --device cuda:0 \
  --num_envs 32 --seed 17 --steps 960 --stage 5 --eval_terrain \
  --contact_observation true --contact_safety_reward true \
  --model_error_history 3 --strict_contact \
  --drive_substep_interpolation true --report /tmp/ppo3_heldout.json

# Evaluate the previous 66x5 production checkpoint (unchanged)
$PY verge_task/scripts/evaluate.py \
  logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt \
  --actuator rom --headless --device cuda:0 \
  --num_envs 32 --seed 4242 --steps 960 --stage 5 --eval_terrain \
  --drive_substep_interpolation true --report /tmp/ppo3_heldout_66.json

# Tests
$PY -m pytest -q verge_task/tests PPO4_ID_FD_PhysX/tests
```

### Varying the held-out mesh

`--seed` does **not** vary a held-out evaluation: the mesh is pinned at seed
1337, the start phase is fixed and the actor is evaluated at its mean, so
every `--seed` returns bit-identical numbers. This was measured, not assumed
(`reports/contact_safe_v1/prefix_identical_seed_artifact/`). Use
`--eval_terrain_seed 909` to move the mesh; never on a gate run.

## Layout

```text
env.py, terrain_env.py           runtime, sensors, contact-safety state machine
env_cfg.py, terrain_env_cfg.py   shared task configuration
actions.py                       action filters and safety controllers
terrain_observations.py          shared observation (66 or 96) and its width helper
terrain_rewards.py               reward terms, incl. the seven contact terms
terrain_convergence.py           held-out gates + strict contact-safety gates
scripts/train.py                 one PPO process
scripts/train_until_converged.py training/evaluation supervisor
scripts/evaluate.py              headless evaluation and GUI
scripts/probe_contact_loss.py    event-level contact-loss diagnosis
scripts/migrate_shared_policy.py checkpoint migration (widen obs, or 54x4 -> 66x5)
scripts/run_*.sh                 diagnosis / sweep / contact-safe training drivers
tests/                           fast contracts
```

PPO is the sole scored controller. The contact-safety state machine observes
the PhysX contact sensor and the cutter's own kinematics and writes no
cylinder target, valve command or joint reference — enforced structurally by
`tests/test_contact_safety.py::test_the_state_machine_writes_no_plant_command`.
