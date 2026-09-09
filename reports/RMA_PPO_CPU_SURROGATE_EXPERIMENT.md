# RMA/PPO CPU surrogate experiment

## Verdict

**NO-GO for Phase 2/3 online adaptation.** A frozen recursive SD model supports a
scientifically bounded **tracking-only** CPU surrogate, and the implemented
authority/lag/delay wrapper changes its response. However, the original PPO did
not degrade in the baseline-only stress sweep, and the short privileged PPO
fine-tune did not learn practically separable context-conditioned actions.
There is therefore no demonstrated PPO adaptation opportunity to justify AAM
training. No result here is equivalent to PhysX or final thesis validation.

The run was CPU-only with the local PyTorch installation (`2.13.0`, CUDA
unavailable). Historical PPO, ID/FD, ROM, and AAM checkpoints were not
overwritten.

## Candidate inventory

| Model | Inputs | Outputs | dt | Recursive? | Context/factors supported |
|---|---|---|---:|---|---|
| `learned_models/rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt` | 6x24 hydraulic history + 8-D desired cylinder command | valve, extension/velocity increments, pressure increment | 50 ms | Yes | old hydraulic `system_factors` option; no mass, q, EE, or contact |
| `learned_models/sd_model/stage31_frozen/z2_rom` | 12-D hydraulic state, 8-D previous/current valve, 10-D q, causal GRU belief, old 6-factor encoder latent | delta hydraulic state (extension, velocity, dP) | 50 ms | Yes | old factor latent; q is held-start in leakage-free rollout; no learned q/EE/contact dynamics |
| `outputs/stage31_rom/sd_none` (selected) | same state/action/q/belief contract, no latent | delta hydraulic state | 50 ms | Yes | no latent; suitable for external authority/lag/delay only; mass is not represented |
| frozen ID | hydraulic state, target motion, previous valve | valve command | — | No | command mapping only; not an environment by itself |

The command ROM was rejected as the environment plant. On held-out test
trajectories its independent recursive errors were already about 2.9 mm,
0.080 m/s, and 21.4 MPa at one step, growing to 43.5 mm, 0.191 m/s, and
39.1 MPa at 20 steps. Optimizing PPO against this drift would be vulnerable to
model exploitation.

The frozen SD gate is materially better. The paired held-out `z2` rollout
(`q_rollout=hold-start`) produced the following true-latent RMSEs:

| Horizon | extension | velocity | dP | all-state |
|---:|---:|---:|---:|---:|
| 1 | 0.213 mm | 0.00740 m/s | 0.504 MPa | 0.291 MPa |
| 10 | 1.718 mm | 0.01467 m/s | 3.112 MPa | 1.797 MPa |
| 20 | 2.687 mm | 0.01720 m/s | 5.134 MPa | 2.964 MPa |
| 50 | 5.350 mm | 0.02058 m/s | 9.313 MPa | 5.377 MPa |

At 50 ticks, all-state RMSE was 5.377 MPa (true), 5.880 MPa (mean),
5.953 MPa (shuffled), and 6.390 MPa (zero). This validates a hydraulic
tracking surrogate and the old SD latent gate; it does not validate the new
three-factor RMA context or a full arm task.

## CPU task and stress contract

`experiments/RMA_PPO/cpu_surrogate_env.py` implements a vectorized torch CPU
environment over frozen 50-ms ROM rows. It preserves the deployed command
semantics:

`PPO action -> 0.004 m cylinder motion increment (validated negative axis sign) -> frozen ID -> valve -> stress wrapper -> SD`.

The reward is explicitly tracking-only:

`-mean((extension error/0.02)^2) - 0.25 mean((velocity error/0.10)^2) - 0.01 mean(action^2)`.

The observation is a finite 66-D PPO-shaped proxy containing hydraulic
tracking channels. q is conditional and held at episode start; no joint,
end-effector, terrain, contact, or full thesis reward is fabricated.

Only reachable stresses were exposed to the policy and plant:

`[actuator_authority_alpha, hydraulic_response_time_tau, actuator_delay_steps]`.

`mass_scale` was excluded because the selected SD transition is not
mass-conditioned. The wrapper checks were positive:

| Check | Result |
|---|---:|
| nominal wrapper max absolute difference | 0.0 |
| authority 0.01 output/input norm ratio | 0.0100000 |
| delay-2 first three outputs for 1,2,3 input | 0, 0, 1 |
| tau=0.10 s first step at dt=0.05 s | 0.393469 |
| fixed-sequence nominal/strong extension separation | 0.002278 m |

## Baseline-only stress gate

The sweep used the original 66-D PPO actor on matched frozen validation
trajectories. The ranges were fixed before the privileged fine-tune:

| Regime | alpha | tau (s) | delay | extension RMSE | return |
|---|---:|---:|---:|---:|---:|
| nominal | 1.0 | 0.00 | 0 | 0.02013 m | -77.72 |
| mild | 0.8 | 0.04 | 1 | 0.01041 m | -23.91 |
| medium | 0.6 | 0.08 | 2 | 0.01213 m | -34.51 |
| strong | 0.5 | 0.12 | 3 | 0.00915 m | -20.77 |

All episodes remained finite, but every stressed error was **lower** than the
nominal error (ratios 0.517, 0.603, and 0.455). Thus the required region
“nominal good, stressed degraded without collapse” was not found. The stress
range is not an adaptation-opportunity curriculum; it is retained only as a
bounded sanity sweep in the artifact JSON.

## Phase 1 sanity fine-tune and gates

For checkpoint surgery, a new 69-D three-context checkpoint was created from
the frozen 66-D PPO. Random-batch equivalence was exact:

`max |pi_old(o) - pi_new(o,c)| = 0.0`.

A four-iteration, one-seed CPU PPO sanity run was performed from that new
checkpoint (not from scratch). Mean iteration returns were -19.68, -9.01,
-9.32, and -4.68; all gradients/parameters stayed finite. The new first-layer
context weight norm reached only 0.00250.

The mandatory matched ceiling was then evaluated with true, mean, shuffled,
and zero policy context while keeping the actual surrogate plant identical:

| Regime | True ext RMSE | Mean | Shuffled | Zero |
|---|---:|---:|---:|---:|
| nominal | 0.017700 | 0.017700 | 0.017700 | 0.017700 |
| mild | 0.011364 | 0.011364 | 0.011364 | 0.011364 |
| medium | 0.010950 | 0.010950 | 0.010950 | 0.010951 |
| strong | 0.010262 | 0.010262 | 0.010262 | 0.010262 |

Differences are at numerical-noise scale and are not consistently ordered in
favor of true context. The action-separation test gave mean
`||a_true-a_wrong|| = 0.000240`, p95 `0.000385`, and 0% of samples above the
declared 0.001 practical threshold. `||a_true-a_mean||` averaged 0.000078.
This is a context-ignoring policy for scientific purposes, so the privileged
ceiling fails.

## Decision and artifacts

Phase 2 on-policy AAM history collection, Phase 2 AAM training, and Phase 3
estimated-context fine-tuning were not run. The correct next step is a future
GPU/PhysX experiment that first creates a genuine baseline degradation region
and uses a transition model with full task state if full arm/contact claims are
required.

New CPU result artifacts are under `experiments/RMA_PPO/cpu_surrogate/`; the
environment and runner sources are `experiments/RMA_PPO/cpu_surrogate_env.py`
and `experiments/RMA_PPO/run_cpu_surrogate_experiment.py`:

- `cpu_surrogate_env.py` and `run_cpu_surrogate_experiment.py`
- `sd_z2_rollout.json` (held-out recursive SD validation)
- `cpu_surrogate_results.json` (stress, baseline, Phase-1, and gate results)
- `phase1_privileged_init_3ctx.pt` and `phase1_privileged_trained_3ctx.pt`

The only new report is this file. The results are surrogate-domain evidence,
not a PhysX-equivalent thesis result.
