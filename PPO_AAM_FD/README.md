# PPO-AAM-FD — the adaptive stack

The thesis calls this stack **AF**. It carries the online-adaptation result.

```text
policy -> valve command -> bounded history-conditioned residual -> frozen FD -> PhysX
```

The policy emits the valve command itself. There is no inverse model in the
command path, and that is the point of the architecture: with one in place the
correction was added to a command already resting on its clip, so an informative
estimate produced no closed-loop effect. Moving the correction onto a command
with headroom is what changed the answer. Thesis §6.7.2 and §5.6.3.

Adaptation enters as an additive bounded residual on that command:

```text
u = clip(u0 + alpha * tanh(g(o, phi(H))), -1, 1),   u0 = frozen_policy(o)
```

`phi` is a temporal-convolution encoder over 20 rows of `(base observation,
previous action)` on the 60 Hz control grid, a 333 ms window carried in the
observation vector rather than in a buffer inside the module. PPO calls the
actor many times per update, so a module-held buffer would be advanced by the
optimiser and desynchronise from the simulation.

Only `phi` and `g` train: 53,836 parameters against a 59,588-parameter frozen
base. `g`'s final layer is zero-initialised and `tanh(0) = 0`, so at
construction the wrapped policy reproduces the frozen base bit for bit. Setting
`alpha` to 0 at evaluation recovers the base exactly, which is what makes the
capacity-matched control a single-variable comparison.

## What it establishes, and what it does not

On the better of two replicating seeds the residual improves cut height by
62.6 % and lateral tracking by 78.6 % against its own frozen base, over 880
paired environments on five held-out meshes. A capacity-matched control with its
history window zeroed does not deliver that gain, so it belongs to the live
history and not to the added parameters.

Two of five training seeds pass both metrics and the median seed does not meet
the pre-registered bar. The result is a demonstrated mechanism, not a reliable
method.

AF is **not** ranked against PPO-ROM or PPO-ID-FD on task error. It does not
share their action contract: they emit a bounded cylinder increment, AF emits
the valve command. The quantity the policy chooses is not the same quantity.

## Frozen runtime models

```text
logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt   the frozen base
id_fd_model/best_fd_cylinder_strict.pt                      forward model
```

The retired latent path additionally needs `outputs/stage31_rom/aam_cnn/L20_d64/aam.pt`
and `outputs/stage31_rom/id_adapter_cnn_l20_s5/zhat_adapter.pt`. Every artifact
above is tracked, with its SHA-256 in thesis Appendix C.

## Train an arm

Every arm in the thesis was trained with this command, differing only in the
seed and, for the control, in `--no_history`.

```bash
export MULAG_DIRECT_VALVE=1 MULAG_RMA_CONTEXT=1
export MULAG_MASS_BIMODAL=1.0,3.0 MULAG_MASS_STEP_TIME_S=20.0 \
       MULAG_MASS_STEP_JITTER_S=8.0 MULAG_MASS_STEP_SCALE=3.0

python PPO_AAM_FD/scripts/train_residual_adapter.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --num_envs 256 --iterations 400 --history 20 --latent_dim 8 --alpha 0.5 \
  --lr 3e-4 --seed 17 \
  --lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/rebal17_history --headless
```

Note the flag asymmetry, which has caused mistakes here: training takes
`--no_history`, scoring takes `--residual_no_history`.

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

## Score

```bash
python PPO_AAM_FD/scripts/evaluate_gates.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/rebal17_history/model_399.pt \
  --residual_alpha 0.5 --residual_history 20 \
  --num_envs 176 --steps 900 --meshes 1337 --seed 17 \
  --report outputs/rif_delay/rebal17_history_m1337.json
```

Evaluations are serial: one at 176 environments holds about 17 GB, and two will
exhaust a 31 GB box. Score the full path; a short window hides ground-contact
failures that appear late along the 26 m line.

## Verify without a simulator

CPU-only. Checks causal warm-up, reset isolation, the bounded residual and the
fallback path.

```bash
python PPO_AAM_FD/scripts/preflight.py --num-envs 4 --device cpu
python PPO_AAM_FD/scripts/minimum_online_adaptation_check.py --device cpu --num-envs 2 --ticks 72
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
this check. See invariants 21 and 22 in `memory.md`.
