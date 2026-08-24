# RMA PPO adaptation: CPU preparation

## 1. Scope and decision

This is a CPU-only preparation and implementation pass. No IsaacLab process,
PhysX simulation, PPO training, rollout collection, GPU execution, or data
collection was run. The final architecture is:

```text
randomized deployment plant -> PPO with privileged context -> on-policy history
                                  |                         |
                                  +-------------------------+-> AAM context estimate
                                                           -> PPO fine-tune
deployed controller: PPO is the sole controller; AAM supplies context only
```

The old six-factor AAM route is not reused. E3, E5, frozen ID/FD checkpoints,
and historical AAM artifacts remain intact.

## 2. Promoted PPO contract traced first

The clean source for Phase 1 is the validated ID-FD checkpoint
`logs/rsl_rl/ppo4_1_sole/speed_range_2_7_pathfix_v2_idfd/model_promoted_2_7kmh.pt`.
The existing 68-D PPO_ID_AAM_FD checkpoint remains a historical artifact and
was not overwritten.

For completeness, the currently promoted online-AAM continuation at
`logs/rsl_rl/ppo_id_aam_fd/train_final/model_599.pt` was also inspected: it is
68-D with actor/critic first layers `(256,68)`, four actions, and actuator
`id_aam_fd`. Its final two columns are the legacy 2-D online AAM contract, so
it is deliberately not the Phase 1 source for the four-coordinate RMA pivot.

| Component | Audited contract |
|---|---|
| PPO environment/config | `PPO_ID_FD.env_cfg.Ppo41IDFDEnvCfg`, inherited PPO3_1 terrain/reward with `cylinder_dynamics_model="fd"`, `use_cylinder_id_pipeline=True` |
| Policy observation | 66 values from `verge_task.observations.build_sole_observation` |
| Action | 4 normalized cylinder commands; action space 4 |
| Action scaling | 0.004 m cylinder travel per unit action, with the existing smoothing/slew/limit chain |
| Checkpoint | `model_promoted_2_7kmh.pt`, iteration 2322, actuator `idfd` |
| rsl_rl actor/critic | ELU MLP 66→256→128→64→4 and 66→256→128→64→1 |
| Actor first layer | `actor_state_dict["mlp.0.weight"]`: `(256, 66)` |
| Critic first layer | `critic_state_dict["mlp.0.weight"]`: `(256, 66)` |
| Normalization | actor and critic `obs_normalizer._mean/_std/_var`, each `(1,66)` |
| Command route | PPO → existing sole-action chain → desired cylinder motion → ID → FD → PhysX; no valve action is exposed to PPO |
| Evaluation entry | `PPO_ID_AAM_FD/scripts/evaluate_gates.py` delegates to `PPO_ROM/scripts/evaluate_gates.py`; `PPO_ID_AAM_FD/visualize.sh` is the canonical wrapper |

The checkpoint reports the same four-axis 0.004 m authority and no teacher
execution. This is the correct base for the new context experiment.

## 3. New context contract

The policy-relevant context is exactly four values:

```text
c_priv = [actuator_authority_alpha,
          hydraulic_response_time_tau,
          actuator_delay_steps,
          mass_scale]
```

`com_z` is optional metadata only. It is not in the default four-wide policy
observation because native COM wiring must be proven first. No old six-factor
coordinate, latent, or classical AAM label is used by this path.

The explicit default normalization is `(mean, std) =
((1.0, 0.08, 1.0, 1.0), (0.5, 0.08, 1.0, 0.25))`. The PPO columns are direct
normalized context values, not a learned encoder output in Phase 1.

## 4. Common deployment-actuation wrapper

`PPO_ID_AAM_FD/rma_adaptation/core.py` adds the opt-in
`DeploymentActuation` wrapper. Its order is:

1. authority: `dx_authorized = alpha * dx_raw`;
2. stable first-order lag: `y[k+1] = y[k] + (1-exp(-dt/tau))*(u-y[k])`;
3. pure FIFO delay in control ticks.

The implementation is batch-safe for four-action tensors, clamps legal
authority/delay values, and resets state per episode. `tau=0` is the explicit
no-lag bypass. At `(alpha,tau,delay,mass)=(1,0,0,1)`, the output is exactly the
raw command. The wrapper stops before ID/FD/ROM/PhysX; it does not create a
second controller or replace the existing action chain.

## 5. Native mass and COM wiring

`NativeRigidBodyWiring` and `verify_native_mass_com_wiring` require mass and
COM to be sourced from native PhysX rigid-body fields. Surrogate observation
channels are rejected. The declaration is:

```text
mass_field = articulation.data.default_mass
com_field  = articulation.data.default_com_pos_b
```

The actual IsaacLab setter remains a future GPU integration step; this CPU pass
only makes the source and the guard explicit.

## 6. PPO privileged-observation extension

Phase 1 appends the four normalized context values to the existing 66-D policy
observation, giving a 70-D actor and critic input. The default/base observation
and four-action contract are unchanged. `rma_adaptation.modes` owns the
assembly so an evaluator cannot silently append a different width.

## 7. Checkpoint surgery

`PPO_ID_AAM_FD/scripts/migrate_privileged_context_checkpoint.py` performs a
non-destructive 66→70 expansion:

- copies actor and critic state dictionaries;
- appends four zero columns to each `mlp.0.weight`;
- appends mean zero, std one, and variance one to both normalizers;
- preserves all old weights and metadata;
- clears the optimizer state because parameter shapes changed;
- records the four context names and the migration provenance.

The generated initialization is
`experiments/RMA_PPO/phase1_privileged_init.pt`. The source checkpoint remains
66-D. A CPU numerical test evaluates the old and converted actors on random
base observations with random context and confirms equal outputs to `1e-6`.

## 8. Phase 1 configuration

`PPO_ID_AAM_FD/rma_adaptation/config.py` and
`experiments/RMA_PPO/config.json` define a conservative, configurable sequence:

```text
nominal -> authority -> lag -> delay -> mass -> combined
```

The prepared ranges are `alpha=[0.25,1.0]`, `tau=[0,0.12] s`,
`delay=[0,3]` ticks, and `mass_scale=[0.85,1.15]`. All Phase 1/2/3 flags are
disabled in the checked-in JSON; enabling them is a future run decision.

## 9. On-policy history schema

`OnPolicyHistoryRecorder` records only PPO-causal rows:

```text
obs_t, action_t, obs_tp1, reward_t, done_t
```

The schema is `rma_on_policy_v1`, has no `true_context`, `plant_id`, or
privileged-label field, and exposes causal mismatch features
`[obs_t, action_t, obs_tp1-obs_t]`. It is intentionally not a loader for the
old classical AAM dataset.

## 10. AAM skeleton and Phase 2 boundary

`AAMEstimator` is a small causal GRU followed by LayerNorm and a four-wide
head. `AAMTrainer` is a CPU synthetic-tensor smoke trainer only. It can be
used to validate tensor shapes and gradients; no historical dataset is read
and no real AAM is trained in this pass. `Phase2Config.source` is explicitly
`ppo_on_policy_rollouts_only` and
`train_on_historical_classical_dataset=False`.

## 11. Phase 3 configuration

Phase 3 is declarative only. The prepared mode is
`AAM_ESTIMATED_FINETUNED`, with the AAM frozen before PPO fine-tuning and the
privileged-versus-mean/shuffled gate mandatory before any fine-tune. PPO remains
the sole actuator controller; AAM output is context, never a valve or cylinder
command.

## 12. Runtime modes and leakage guards

The exact runtime enum is:

```text
BASELINE, PRIVILEGED, MEAN_CONTEXT, SHUFFLED_CONTEXT,
AAM_ESTIMATED, AAM_FROZEN, AAM_ESTIMATED_FINETUNED
```

`assemble_policy_observation` appends exactly one four-wide source (or none for
baseline). `assert_no_context_leakage` rejects the four privileged names,
`com_z`, `true_context`, `privileged_context`, and `plant_id` in the AAM feature
schema. The shuffled arm changes context routing only; it cannot alter the
actuation wrapper or the PPO action path.

## 13. Future opportunity sweep and analysis

`delta_adapt(robust, privileged)` implements
`Delta_adapt = E_robust - E_privileged`. `recovery_fraction` reports the
fraction recovered before a declared threshold, and `stress_thresholds`
declares the authority, lag, and delay onset values. The future sweep should
report the identity point, α=0.01 positive-control point, α=0, independent
one-factor stages, and combined stresses with identical seeds and episodes.

## 14. Mid-episode switching and recovery

The future evaluator must switch context at a declared mid-episode tick and
report recovery time, recovered fraction, tracking error, action continuity,
and termination. The seven modes above must be evaluated under the same plant
draws. A context estimate is not promoted because it improves a reset-only
average; it must recover after a within-episode deployment change and pass the
privileged-versus-mean/shuffled gate.

## 15. CPU evidence and files

The required local PyTorch installation is available at
`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`:

```text
torch 2.13.0, CUDA available: False, device count: 0
```

`PPO_ID_AAM_FD/tests/test_rma_adaptation_cpu.py` contains 16 CPU tests. The
repository image has no pytest module, so the same 16 assertions were executed
by a direct Python 3.11 harness; all 16 passed. Source compilation and import
checks also passed. No simulator or PPO process was launched.

## 16. GPU runbook and promotion gate

`experiments/RMA_PPO/runbook.sh` contains the ordered future GPU steps 0–10:
environment/artifact checks; nominal identity smoke; six staged Phase 1
training arms; mandatory privileged, mean, and shuffled evaluation; on-policy
recording; AAM training/evaluation; Phase 3 fine-tuning; mid-episode recovery;
analysis; promotion; and final hashes.

The mandatory order is intentional: do not train or promote an AAM until the
privileged context beats both mean and shuffled controls on the declared robust
gate. The runbook is documentation for a later GPU session and was not run in
this CPU preparation pass.
