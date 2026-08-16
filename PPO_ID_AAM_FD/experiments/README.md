# PPO-ID-AAM-FD — adaptation experiments

Experiments testing the pipeline's central claim:

> The AAM estimates a latent of the current hydraulic-mechanical dynamics from
> interaction history. That latent conditions both the task policy and the
> inverse-dynamics model, enabling adaptation at the policy and actuator-mapping
> levels.

$$\hat z_t=\phi(h_t),\qquad a_t=\pi(o_t,\hat z_t),\qquad u_t=ID(x_t,a_t,\hat z_t)$$

The controller never receives $\theta$. Only the plant knows the true dynamics;
the AAM must infer their effect from history. FD is deliberately **not**
conditioned on $\hat z$, so an improvement cannot be confused with simply making
the simulated plant more accurate.

## What is already wired

| component | status |
|---|---|
| $h_t=[q,\dot q,\Delta P,u]_{t-H:t}$ → AAM | `feature_contract: joint_q + joint_qdot + dP + previous_valve`, 18 × 20 lag |
| $a_t=\pi(o_t,\hat z_t)$ | `online_runtime.py: _apply_latent_mode` |
| $u_t=ID(x_t,a_t,\hat z_t)$ | `online_runtime.py: condition_valve` concatenates `z_hat` into adapter features |
| FD **not** conditioned | FD takes $(x,u)$ only |
| the 2×2 ablation | two independent flags — `online_aam_latent_mode` (PPO) × `online_aam_controller_latent_mode` (ID) |

The ablation table maps directly onto existing flags:

| controller | PPO gets $\hat z$ | ID gets $\hat z$ | `--online_aam_latent_mode` | `--online_aam_controller_latent_mode` |
|---|---|---|---|---|
| Baseline | ✗ | ✗ | `zero` | `zero` |
| ID adaptation | ✗ | ✓ | `zero` | `estimated` |
| PPO adaptation | ✓ | ✗ | `estimated` | `zero` |
| Full adaptation | ✓ | ✓ | `estimated` | `estimated` |

**Caveat:** `AAM_module/tools/promote_system.py:223` hard-requires
`controller_latent_mode == "estimated"`, which rejects two of the four cells.
Relax that guard before running the ablation through the promotion gate.

## Why step 1 comes first

Training already randomizes $\theta$ — domains `combined` (64 runs),
`hydraulic` (40), `mechanical` (12), `nominal` (2), spanning flow, compliance,
seal, pressure, mass_scale and com_z. So the usual failure mode
("one fixed plant ⇒ $\hat z\approx$ const") does **not** apply here.

But decoding the true factors from the deployed latent on held-out runs gives:

| factor | held-out R² |
|---|---|
| seal | **0.956** |
| compliance | 0.448 |
| flow | 0.114 |
| pressure | 0.081 |
| com_z | 0.056 |
| **mass_scale** | **−0.008** |

Mass is randomized during training and still absent from the latent. The
deployed $\hat z$ is 2-D because it regresses onto the frozen SD latent, and SD
is 2-D by construction (extension + velocity channels) —
`train_stage31_aam.py:197` reads `latent_dim = int(y_train.shape[1])`.

**Experiment 1 in the plan is EE mass. Running it against the current latent
would produce a null for a structural reason, not a scientific one.** Step 1
settles that first.

## Experiments

### `latent_width_ceiling.py` — step 1

Separates two explanations for the missing factors:

- **width** — 2 numbers cannot carry 6 factors, or
- **target** — 2 numbers could, but regressing onto the SD latent spends them
  on the wrong thing.

Same causal-CNN trunk and history contract as the deployed AAM; only the
bottleneck changes:

```
history h_t -> CNN trunk -> z (d_z) -> LINEAR head -> theta_hat (6)
```

The head is linear on purpose: PPO and ID consume $z$ directly, so the factors
must be linearly legible *in $z$ itself*, not merely recoverable by a deep
decoder.

Reading the result:

- `d_z=2` trained on $\theta$ is the **target control**. If it beats the
  deployed 2-D latent, the training target is the problem, not the width.
- `d_z>=6` gives the **information ceiling** — how much of $\theta$ is
  recoverable from history at all. If mass_scale stays near zero even at
  `d_z=12`, the limit is the history contract, not the bottleneck.

CPU only, no simulator.

```bash
python3 PPO_ID_AAM_FD/experiments/latent_width_ceiling.py \
    --widths 2 4 6 8 12 --epochs 60 --stride 5
```

Outputs `results/latent_width_ceiling.json` and `.log`.

$\theta$ is constant within a run, so windows are highly redundant and
`--stride` subsamples them; run diversity (82 train runs) carries the probe,
not window count.

### Planned

- **Experiment 1 — EE mass.** Train $m_{EE}\in[0.8,1.2]m_0$, test
  $\{0.7,0.8,1.0,1.2,1.3\}m_0$ for interpolation and extrapolation, across the
  2×2 ablation. Gated on step 1.
- **Experiment 2 — in-episode dynamics change.** $m_{EE}: m_0\to1.3m_0$ at
  $t=8$ s, or $\tau: 20\to60$ ms; plot $\hat z_t$ against tracking error and
  valve command. This is the strongest demonstration and currently has **no**
  supporting code — nothing changes $\theta$ mid-episode yet.
- **Experiment 3 — valve / nonlinearity variation.**

Combine perturbations only after each is understood alone.
