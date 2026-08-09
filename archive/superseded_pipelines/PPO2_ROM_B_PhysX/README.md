# PPO2-ROM_B robust policy

PPO2-ROM_B keeps ROM-A as the immutable nominal reference and trains the
42-observation, four-action PPO2-A policy across an audited hydraulic plant
family. True factors are private runtime state and never enter actor or critic
observations.

## Uncertainty set

Each environment independently samples one factor vector at reset and holds it
fixed for the episode:

```text
e_comp   ~ U(-0.125, 0.020)
d_seal,j ~ U(0, 1), j=1,...,4
```

The nominal reference is `[0, 0, 0, 0, 0]`. Nominal mode bypasses the
factor argument completely and is bit-identical to ROM-A.

Factors modify only explicit gray-box channels:

```text
bulk_B       = bulk_A exp(e_comp)
w_j          = 0.25 d_seal,j
leakage_B,j  = leakage_A,j (1 + w_j)^3

pressure_rate = (4 bulk_B / volume)
                (flow_B - area*velocity - leakage_B*pressure)
```

Pressure then drives force, acceleration, velocity, and extension inside the
ROM. Final motion outputs are not multiplied by the factors. Pressure-limit,
valve-time-constant, and mechanical-clearance randomization are excluded.

## Robustness scale

`h in [0.4, 1.2]` is a dimensionless correlated evaluation coordinate, not the
PPO training distribution. Training samples all five factors independently.

For `0.4 <= h <= 1`:

```text
s = (1-h)/0.6
e_comp = -0.125s
d_seal,1:4 = s
```

For `1 < h <= 1.2`:

```text
s = (h-1)/0.2
e_comp = 0.020s
d_seal,1:4 = 0
```

Thus `h=1` is ROM-A, `h=0.4` is the supported degraded corner, and `h=1.2`
is the supported favourable corner. Evaluation should report the h-curve plus
independent random combinations and one-factor sweeps.

Representative physical multipliers are:

```text
h     stiffness=exp(e_comp)   leakage=(1+0.25d_seal)^3
0.4          0.8825                    1.9531
0.7          0.9394                    1.4238
1.0          1.0000                    1.0000
1.2          1.0202                    1.0000
```

`h` and the resulting factors are never applied to the PPO, ROM outputs,
pressure, velocity, or extension as direct multipliers. They alter compliance
and seal leakage before the existing pressure calculation. The
pressure calculation follows the contract:

```text
pressure_rate = (4*bulk/volume)
                * (flow - area*velocity - leakage*pressure)
```

## Training curriculum

The policy progresses through nominal ROM-A, mild randomization, medium
randomization, and the complete ROM-B uncertainty set. At curriculum severity
`c`, each audited range is contracted about the nominal zero factors:

```text
e_comp   ~ U(-0.125c, 0.020c)
d_seal,j ~ U(0, c), j=1,...,4
```

The four stages are `c = 0.0, 0.4, 0.7, 1.0`. Stage 1 is therefore exactly
nominal even though the runtime remains in random-training mode. Factors are
sampled independently per environment and held fixed until reset. The accepted
`model_607.pt` passed three consecutive full Stage-1 audits and full Stage-2
and Stage-3 audits. Production training therefore starts at Stage 4 (`c=1.0`),
the first unsupported group, instead of degrading behavior that is already
validated at nominal, mild, and medium randomization.

After each stochastic PPO training round, the actor mean is evaluated in a
separate deterministic rollout on all five shapes. The evaluation uses 128
environments, independently samples factors at the current curriculum scale,
and collects 120 scored steps per shape, matching the accepted four-cylinder
promotion contract. Each shape uses a fixed held-out seed
across every round, so trajectory phases and normalized factor draws form a
stable comparison suite. Safety failures are captured per environment without
allowing automatic resets to hide them. Exploratory training actions are never
used as promotion evidence.

A stage advances only after three consecutive deterministic five-shape
evaluations pass every gate:

```text
max shape mean tracking error       <= 0.040 m
max shape p95 tracking error        <= 0.080 m
max shape p95 tool acceleration     <= 40.0 m/s^2
minimum head alignment              >= 0.980
maximum absolute clearance error    <= 0.200 m
maximum post-constraint truck drift <= 0.001 m
maximum safety termination rate     <= 0.05
```

This matches the accepted four-cylinder promotion contract: at least 95% of
replicas must complete, and quality metrics are computed over completed
replicas. Termination rates still include every failed or incomplete replica.

At full randomization, training must run for at least five complete rounds and
pass the same deterministic gates for three consecutive rounds before it stops.
Otherwise it continues until the maximum round budget is exhausted. Decisions
and per-shape metrics are saved in `rom_b_curriculum_history.json`; the final
outcome is also recorded in the run contract.

ROM-B adds bounded tail-risk shaping between soft margins and the exact gates,
plus a fixed event penalty when head, clearance, cylinder-limit, truck, or
acceleration limits are crossed. This avoids the old unbounded quadratic safety
cost while directly aligning PPO returns with promotion. All four action
networks are trainable; actor normalization and exploration standard deviations
remain frozen to the accepted PPO2-A checkpoint. A stage that does not advance
within 12 rounds stops with `stage_stalled` for audit.

Before the first update, the trainer evaluates and snapshots the accepted
policy. Shapes that fail receive three times the PPO iterations in the next
round. A candidate policy is retained only when its normalized gate score
improves; otherwise actor, critic, and optimizer are restored to the best
accepted snapshot. The full-scale audit identifies `line` as the initial focus.

The curriculum severity `c in [0,1]` is distinct from the documented
`h in [0.4,1.2]` correlated evaluation curve. Training uses `c`; robustness
plots use `h`.

## Starting training

Training starts from the selected PPO2-A `model_607.pt`, restores actor, critic,
and their 42-D normalizers, and uses a fresh optimizer. The default command
allows up to 500 updates over five balanced trajectory shapes with 128
environments; it can stop early only after full-scale acceptance:

```bash
cd /home/rrlab/rrlab_mulag
./PPO2_ROM_B_PhysX/train.sh
```

Equivalent explicit command:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/train.py \
  --headless --device cuda:0 --num_envs 128 --seed 17 \
  --balanced_shape_rounds 50 --iterations_per_shape 2 \
  --start_curriculum_stage 4 --curriculum_eval_steps 120 \
  --max_stage_rounds 12 --failed_shape_iteration_multiplier 3
```

Outputs are written under `logs/rsl_rl/ppo2_rom_b_robust/<timestamp>/`. Each
run contains `ppo2_rom_b_run_contract.json` recording the exact factor ranges,
curriculum, factor isolation, checkpoints, trajectories, and h-scale semantics.
The history records deterministic evaluation seeds, completion rates, sampled
factor minima/means/maxima, per-shape metrics, failed checks, and stage changes.

## Resuming an interrupted accepted run

Pass both the accepted checkpoint and its source run directory. Resume restores
actor, critic, optimizer, checkpoint iteration, curriculum history, stage-round
count, and consecutive-pass count. It re-evaluates the checkpoint before using
it as the rollback baseline. Accepted candidates are also saved with an
`_accepted.pt` suffix. `SIGINT`, `SIGTERM`, and terminal hangup request a
graceful stop at the next safe boundary and produce an `interrupted` outcome.

```bash
./PPO2_ROM_B_PhysX/train.sh \
  --resume_checkpoint /absolute/path/to/model_N.pt \
  --resume_run_dir /absolute/path/to/source-run
```

For an h-sweep evaluator, set `cfg.rom_b_mode = "health"` and assign
`cfg.rom_b_health_scale` to each value from `0.4` through `1.2`. Do not use
health mode for the main PPO training run.

## Held-out robustness evaluation

ROM-B uses
`plant_reachable_trajectory_library_v3_circle_c7_retime.json`. The circle tool
geometry and inner controller are unchanged; only the reference timing slows
the mechanically difficult region and recovers speed elsewhere. Model 53 is
certified on the termination gate. The remaining circle acceleration limitation
is documented rather than hidden. See `reports/circle_c7_revision_report_2026-07-26.md`
and `reports/ROM_B_CIRCLE_COMPLETE_TECHNICAL_HANDOFF_2026-07-26.md`.

The dedicated evaluator performs no PPO updates. It runs five fresh independent
full-range Monte Carlo seed groups, then evaluates the correlated health path at
`h = 0.4, 0.5, ..., 1.2` over three matched held-out seed groups. It writes JSON
and CSV rollout data, a Markdown gate report, and a final robustness figure under
`PPO2_ROM_B_PhysX/reports/evaluation_<timestamp>/`.

```bash
cd /home/rrlab/rrlab_mulag
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/evaluate.py \
  --headless --device cuda:0 --num_envs 128 \
  --monte_carlo_rounds 5 --h_sweep_rounds 3 --eval_steps 120
```

The Monte Carlo seeds begin at `seed + 300000`, separate from curriculum
promotion seeds. The h sweep begins at `seed + 400000` and uses the same seed
for a given trajectory shape and seed group at every h value. The five h-derived
factors are installed immediately after reset and before the first observation
or ROM step; h is never applied to policy actions or ROM outputs. The final
figure includes every acceptance gate, per-trajectory means, repeated-seed
worst-shape envelopes, pass/fail markers, completion, and Monte Carlo
termination distributions.

## Offline verification

```bash
PYTHONPATH=/home/rrlab/rrlab_mulag:/home/rrlab/rrlab_mulag/ROM-cylinders_pipeline \
  /home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  PPO2_ROM_B_PhysX/tests/test_rom_b_contracts.py
```

## Visualization

The player defaults to one complete `A -> B -> A` episode. It shows the blue
reference path in Isaac Sim and clean per-episode Matplotlib tracking and
acceleration plots. Independent random factors remain fixed within an episode
and are resampled at reset.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  /home/rrlab/rrlab_mulag/PPO2_ROM_B_PhysX/scripts/play.py \
  --device cuda:0 --trajectory_shape circle --randomized_plant \
  --episodes 1 --realtime
```

The GUI player uses an explicit fixed-chassis fixture because truck motion is
outside the arm-only PPO problem. It prints chassis translation and rotation
errors on every diagnostic line. This visualization fixture is not part of the
earlier certification environment and must not be presented as new certification
evidence without rerunning the evaluator under the same contract.
