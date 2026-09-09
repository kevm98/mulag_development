# PPO-AAM-FD wide verge task

The thesis calls this stack **AF**. PPO emits the valve command itself; there is
no inverse model in the command path:

```text
policy -> valve command -> frozen FD -> PhysX
```

The policy reads a 70-wide observation: 66 task columns plus the 4-wide RMA
context `(alpha, tau, delay, mass_scale)`. Two environment variables select this
stack and must be exported for every command below:

```text
MULAG_DIRECT_VALVE=1    the policy emits the valve command
MULAG_RMA_CONTEXT=1     append the 4-wide context, observation 66 -> 70
```

Frozen base policy, the subject of every run here:

```text
logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
id_fd_model/best_fd_cylinder_strict.pt                     forward model
```

AF is **not** ranked against PPO-ROM or PPO-ID-FD on task error. They emit a
bounded cylinder increment, AF emits the valve command; the quantity the policy
chooses is not the same quantity.

## Trajectory shapes

The cutter's tracked white line, not the truck path. Eleven families, defined as
`TRACKING_PATH_FAMILIES` in `verge_task/terrain.py` and re-exported by
`PPO_ROM/path_shapes.py`:

1. `straight`
2. `lateral_sine`
3. `s_curve`
4. `random_spline`
5. `offset_return`
6. `boom_sweep`
7. `lateral_chirp`
8. `double_s_curve`
9. `lane_change_hold`
10. `gaussian_slalom`
11. `fourier_meander`

The order is load-bearing: each environment is assigned
`TRACKING_PATH_FAMILIES[(env_index + offset) % 11]`, so a run with eleven or
more environments covers every family deterministically.

## Payload

`MULAG_PAYLOAD_RANGE="lo,hi"` sets the payload scale, valid over 1.0--5.0. Pin a
single level with `X,X`. There is no `--payload` flag; it is an environment
variable read by `PPO_AAM_FD/env_cfg.py`.

The payload is drawn **at the start of the episode**, in
`_latch_deployment_factors()` on reset. By default the draw is seeded without
the reset index, so it is a fixed property of each environment and does not
change between episodes; `MULAG_PAYLOAD_RESAMPLE=1` opts into a fresh draw each
episode. It never changes *within* an episode -- RMA treats the context as
constant for the episode.

A mid-episode change is a different knob: `MULAG_MASS_STEP_TIME_S` with
`MULAG_MASS_STEP_SCALE` makes the tool body `baseline * scale` from that instant
onward, and `MULAG_MASS_STEP_JITTER_S` scatters the instant per environment so a
non-adaptive arm cannot compensate on a timer.

## Visualize

`MESH`, `ENVS` and `STEPS` are environment variables, not flags.
`--trajectory-shape` sets the cutter's white line, not the truck path.

**AF frozen** -- the frozen base alone. This is the `af_base` arm of the Chapter 7
runs:

```bash
cd ~/rrlab_mulag
export MULAG_DIRECT_VALVE=1 MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
MESH=1337 ENVS=4 STEPS=960 PPO_AAM_FD/visualize.sh \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --trajectory-shape double_s_curve
```

Sweep the payload one level per process, never in one run -- pinning it per
process is what keeps the levels comparable:

```bash
for P in 1 3 4 5; do
  export MULAG_DIRECT_VALVE=1 MULAG_RMA_CONTEXT=1
  export MULAG_PAYLOAD_RANGE=${P}.0,${P}.0
  MESH=1337 ENVS=4 STEPS=960 PPO_AAM_FD/visualize.sh \
    logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
    --trajectory-shape double_s_curve
done
```

**AF adaptive** -- the `af_adaptive` arm adds the trained history-conditioned
residual on top of the same frozen base:

```bash
export MULAG_DIRECT_VALVE=1 MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
MESH=1337 ENVS=4 STEPS=960 PPO_AAM_FD/visualize.sh \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/rebal17_history/model_399.pt \
  --residual_alpha 0.5 --residual_history 20 \
  --trajectory-shape double_s_curve
```

The capacity-matched control (`af_control`) is the same command against the
no-history arm, with its history window zeroed:

```bash
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/rebal17_nohistory/model_399.pt \
  --residual_no_history --residual_alpha 0.5 --residual_history 20
```

`--residual_alpha 0` recovers the frozen base exactly, which is what makes the
control a single-variable comparison. The residual modules and both arm
checkpoints were restored from `d99f319^` and `35b2e7f^`; `train_residual_adapter.py`
was **not** restored, so these arms can be scored but no new arm can be trained.

Close the Isaac window to stop.

## Train

Trains the RMA context channel from the frozen base; the base weights are held.

```bash
PY=~/miniconda3/envs/env_isaaclab/bin/python
export MULAG_DIRECT_VALVE=1 MULAG_RMA_CONTEXT=1
export MULAG_MASS_BIMODAL=1.0,3.0 MULAG_MASS_STEP_TIME_S=20.0 \
       MULAG_MASS_STEP_JITTER_S=8.0 MULAG_MASS_STEP_SCALE=3.0

$PY PPO_AAM_FD/scripts/train.py \
  --resume logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --train_context_only \
  --num_envs 256 --stage_iterations 400 --seed 17 \
  --lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/my_af_run --headless
```

## Score

```bash
$PY PPO_AAM_FD/scripts/evaluate_gates.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --num_envs 176 --steps 900 --meshes 1337 --seed 17 \
  --report outputs/rif_delay/af_base_m1337.json
```

Evaluations are serial: one at 176 environments holds about 17 GB, and two will
exhaust a 31 GB box. Score the full path; a short window hides ground-contact
failures that appear late along the 26 m line.

## Verify without a simulator

CPU-only. Checks causal warm-up, reset isolation and the fallback path.

```bash
$PY PPO_AAM_FD/scripts/preflight.py --num-envs 4 --device cpu
$PY PPO_AAM_FD/scripts/minimum_online_adaptation_check.py --device cpu --num-envs 2 --ticks 72
```

## Before trusting any result

`MULAG_MASS_BIMODAL=1.0,X` assigns **half the environments to each mode from
spawn**. It is not a purely mid-episode change, and raising X makes the plant
statically heavier as well as making the flip larger. For a uniform step, omit
`MULAG_MASS_BIMODAL` entirely so `MULAG_MASS_STEP_SCALE` applies to every
environment. Setting it to `1.0,1.0` does not work: both modes are 1.0, the
bimodal path wins, and the mass never moves.

Always confirm the plant did what you asked by reading `mass_scale_mean` from
the trace before analysing. One dose-response reading was corrected for want of
this check.
