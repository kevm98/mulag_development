# RMA on a clean hydraulic cylinder model — design

Written 2026-08-29. Supersedes the plant recommendation in
`RMA_ARCHITECTURE_DESIGN.md`, which proposed running RMA on the ROM *as
trained*. This document says what to run it on instead, and why the difference
matters. `AAM_RMA_REPAIR.md` still holds for gate order and thresholds.

---

## 1. You already have the clean hydraulic cylinder model

The thesis states the plant equations in Chapter 3. Chamber continuity
(`eq:sys:continuity`):

$$\dot{p}_1 = \frac{E_{\text{eff}}}{V_1(x)}\bigl(Q_1 - A_1\dot{x} - C_{\text{leak}}(p_1-p_2)\bigr)$$

with effective bulk modulus for entrained air (`eq:sys:bulk`)
$E_{\text{eff}}(p) = (1/E_f + \alpha/p)^{-1}$, orifice flow depending on spool
position and $\sqrt{\Delta p}$, spool lag $\dot{x}_v = (u - x_v)/\tau_v$, and
piston force $F_h = p_1A_1 - p_2A_2 - F_{\text{fric}}(\dot{x})$.

`learned_models/rom_model/ROM_pipeline/models/hydraulic_rom.py` implements that structure
directly, at lines 216--234:

```python
pressure_margin = softplus((supply - smooth_sign * pressure) / 1e6) * 1e6
flow          = actuation * flow_gain * sqrt(0.5 * pressure_margin)      # orifice
pressure_rate = (4 * bulk / volume) * (flow - area*velocity - leakage*pressure)  # continuity
force         = next_pressure * area + force_residual
friction      = viscous_friction * velocity
acceleration  = (force - friction - load) / mass                          # Newton
```

with eight **physically named, trainable** parameters per cylinder:

| parameter | init | clamp | physical meaning |
|---|---|---|---|
| `log_bulk_modulus` | $1.5\times10^{9}$ | $[10^{7}, 3\times10^{9}]$ | $E_{\text{eff}}$, fluid + entrained air |
| `log_leakage` | $10^{-12}$ | $[10^{-15}, 10^{-7}]$ | $C_{\text{leak}}$, internal leakage |
| `log_supply_pressure` | $3.4\times10^{7}$ | $[10^{6}, 5\times10^{7}]$ | $p_s$ |
| `log_flow_gain` | $10^{-8}$ | $[10^{-9}, 10^{-3}]$ | orifice coefficient |
| `log_effective_area` | $6.4\times10^{-3}$ | $[10^{-4}, 5\times10^{-2}]$ | $A$ |
| `log_nominal_volume` | $10^{-3}$ | — | $V_0$, dead volume |
| `log_effective_mass` | $10^{7}$ | $[10, 10^{7}]$ | reflected inertia |
| `log_viscous_friction` | $5\times10^{4}$ | $[1, 10^{7}]$ | $F_{\text{fric}}$ coefficient |

This is the clean model you are asking for, it is already differentiable, and it
already runs at RL speed. Nothing needs to be written from scratch.

## 2. Why it does not currently behave like one

The parameters degenerated during training. The code says so in its own comment
at line 108 --- *"conservative initializations and must be replaced or bounded
using the dataset audit"* --- and that audit was never applied. In the deployed
checkpoint:

- `effective_mass` and `viscous_friction` sit at their $10^{7}$ **ceilings**;
- `effective_area` sits at its $10^{-4}$ **floor**.

Read those back into the equations. With $m = 10^{7}$, $a = (F - F_v - L)/m
\approx 0$: the physical velocity path produces nothing. With $A = 10^{-4}$,
$F = pA$ is negligible, so the pressure state barely couples to force either.
The physics branch is numerically switched off, and the E5 Stage-2 audit
confirms the consequence empirically: **with the residual heads disabled the
model does not move at all** --- zero travel, zero extension error, at every
stress level. $100\%$ of realised displacement originates in the learned
residual heads.

So the current ROM is a black box wearing a physics costume. Randomising
`seal` or `compliance` moves `leakage` and `bulk` in an equation whose output is
then swamped by `pressure_residual`, and the visible effect on displacement is
whatever `factor_response_gain` --- a hand-written scalar clamped to
$[0.35, 1.50]$ --- imposes. That clamp, not the hydraulics, is what E5 measured
as the $62\%$ authority bound.

**This is the single repair that makes everything else meaningful.** A latent
over hydraulic parameters can only be identifiable from a trajectory if those
parameters actually shape the trajectory.

## 3. Why keeping FD as-is cannot give you clean hydraulics

`CylinderFDRuntime.step(valve_cmd, extension, extension_velocity, dP, force)`
(`cylinder_models_runtime.py:320`) takes **no parameter argument**. It is a
frozen learned map from valve and state to next state --- one fixed plant. It
cannot express a leakage change, a bulk-modulus change or a supply-pressure
change, because it has nowhere to put them.

That is exactly why E3 hit its bind: `z_hat` decodes seal at $0.956$, and seal
is inert on the plant AIF actually runs. FD is not the clean hydraulic model; it
is a *single sample* from the family of hydraulic models, with the parameters
baked in and unreachable.

To keep FD in the loop you would have to retrain it as
$\text{FD}_\theta(\text{valve}, x, \dot{x}, \Delta p, F, \mathbf{e})$ --- a
parameter-conditional forward model, trained over the randomisation design. That
is a legitimate option (§4, P2) but it is strictly more work than repairing the
ROM, and it yields a *learned* parameter dependence rather than a structural
one.

## 4. Plant options

| | plant | parameters act via | effort | fidelity of the parameter dependence |
|---|---|---|---|---|
| **P1** | ROM with repaired physics | the equations themselves | **medium** | structural --- exact |
| P2 | factor-conditional FD | learned input | high | learned --- approximate |
| P3 | analytic ODE written fresh | the equations | high | structural, but discards the identified fit |

**Recommendation: P1.** It reuses the identified model, keeps the deployment
interface unchanged, and makes the parameter dependence structural rather than
learned. P3's only advantage over P1 is a clean slate, which costs you the
Simscape identification you already paid for.

### P1 in detail — the repair

1. **Fix the physically knowable parameters** to audited values instead of
   learning them. $A_1, A_2$ and $V_0$ are geometry: take them from the machine,
   not from gradient descent. $p_s$ is a measured operating pressure. This alone
   removes the area degeneracy.
2. **Re-identify the remaining parameters** ($E_f$, $\alpha$, $C_{\text{leak}}$,
   flow gain, friction, reflected mass) on the Simscape data, with a prior
   penalty toward physically plausible values and a hard penalty on any
   parameter within $5\%$ of a clamp bound. A parameter at its bound is a
   modelling failure, not a fit.
3. **Cap the residual heads.** Add a loss term on residual share so the learned
   terms correct the physics rather than replace it. Target: physics path
   carries the majority of $\Delta x$ at nominal. The existing E5 Stage-2 probe
   (`rom_absorber_axes.json`, `residual_gain` axis) already measures this, so
   the gate exists.
4. **Delete `factor_response_gain`.** With a live physics path the factors act
   through `bulk`, `flow_gain`, `supply` and `leakage` as the equations
   intend. The hand-written gain was a workaround for the dead physics path and
   becomes both unnecessary and a confound.

**Acceptance for the repair, before any RL:** with residual heads disabled the
model must reproduce Simscape step responses to within a stated tolerance, and a
sweep of $C_{\text{leak}}$ and $E_{\text{eff}}$ must change realised
displacement monotonically without touching `factor_response_gain`.

## 5. The randomisation vector

Randomise the **physical parameters**, not the abstract six factors. This is
what makes the latent a plant-identity code rather than an index into a manifest.

| symbol | parameter | suggested range | justification |
|---|---|---|---|
| $C_{\text{leak}}$ | internal leakage | $\times[1, 10^{3}]$ of nominal | wear; the dominant ageing mode |
| $E_{\text{eff}}$ | bulk modulus | $[0.5, 2.0]\,\text{GPa}$ | ch3 validation swept exactly this |
| $p_s$ | supply pressure | $\pm 15\%$ | pump droop, relief setting |
| $K_q$ | orifice/flow gain | $\pm 25\%$ | valve wear, spool tolerance |
| $\tau_v$ | spool time constant | $[1, 3]\times$ nominal | valve dynamics |
| $F_{\text{fric}}$ | seal friction | $[0.5, 3]\times$ | seal wear, temperature |
| $m_{\text{tool}}$ | tool mass | $[0.8, 1.5]$ | existing, acts via PhysX |
| $c_{\text{om},z}$ | tool COM height | $[-0.10, 0.05]$ | existing, acts via PhysX |

The ch3 validation gives the precedent for the bulk-modulus range: sweeping
$\beta \in \{0.5, 1.0, 1.5, 2.0\}$ GPa moved slew delay $0.85 \to 0.65$ s and
tracking error $\approx 11\%$, saturating past $1.5$ GPa. **That is a measured,
task-relevant plant effect** --- the one thing the six-factor design never
demonstrated. Start there: it is the axis most likely to make adaptation pay.

Sample $\mathbf{e}$ per episode, log-uniform for the multiplicative parameters.

## 6. Architecture

```
        e (8 physical parameters, sampled per episode)
        |
        v
   mu(e) -> z, d_z = 4..8        <- trained WITH the policy under PPO
        |
        v
 x --> pi(x, z) --> desired cylinder motion --> HydraulicROM(e) --> PhysX
        ^                                        (physics live)
        |
 z_hat = phi(history)            <- phase 2, regressed onto the same mu
```

Two departures from the current build, both deliberate:

- **`z` is trained with the policy, not with `SD_z`.** In RMA the latent is
  *whatever encoding changes the return*. Training it against a dynamics
  predictor is what produced a seal-dominated code with $0.43\%$ static
  plant-identity content in the closed loop.
- **`d_z` rises from 2 to 4--8.** Two dimensions were chosen to bottleneck six
  abstract factors for SD reconstruction. With eight physical parameters and a
  control objective, 2 is a hard bottleneck. Kumar et al. use 8. Start at 8, and
  reduce only if per-axis reconstruction shows spare capacity.

Keep the ID out of the command path. The ROM is the plant, the policy commands
it directly, and Gate 0's $80\%$ saturation --- a property of the ID→FD path ---
does not arise.

## 7. Training procedure, step by step

Following Kumar et al. (`kumar2021rma`, cited at `ch2:239`) and the A-RMA
extension, with this system's specifics filled in.

### Phase 1 — privileged policy

- **Plant:** repaired ROM + PhysX. `e` sampled per episode from §5.
- **Inputs:** $\pi(x_t, a_{t-1}, z_t)$ with $z_t = \mu(e)$; $\mu$ a small MLP
  trained jointly by PPO. `z` enters **both actor and critic**; LayerNorm on the
  `z` columns; obs-normaliser refit including them.
- **Auxiliary loss:** $\lVert \text{dec}(z) - \mathbf{e} \rVert^2$ so axes carry
  physical meaning and phase 2 has something per-axis to check.
- **Data:** on-policy PPO rollouts. No offline dataset.
- **Hyperparameters:** normal PPO schedule --- *not* the deployed run's
  $3\times10^{-6}$ / entropy $0.001$, which is a fine-tune configuration.
- **Curriculum:** the existing `runbook.sh` staged approach (nominal → one
  parameter at a time → combined) is sound; keep it.

**Gate 1 (blocking).** Score `true z` against `mean z` and `zero`. The privileged
policy must beat the content-free control with a bootstrap 95% CI excluding
zero. E4's $-3.4\%$ with the CI through zero is a **fail**. If it fails, `e` does
not change the optimal action in this regime --- fix the plant or the operating
point, do not train an estimator.

### Phase 2 — adaptation module

- **Freeze** $\pi$, $\mu$, and the plant.
- **Data collection:** run the phase-1 policy **with privileged `z`** in the
  closed loop, logging $(x_t, a_t, z_t)$. This is on-policy, in the deployment
  loop, at deployment speeds and trajectories. It is *not* the existing 2.3 GB
  PRBS collection, which is classical-controller excitation and the wrong
  distribution --- the mismatch Stage 26's offline numbers concealed.
- **Objective:** $\lVert \phi(H_t) - z_t \rVert^2$, plus
  $\lVert \text{dec}(\phi(H_t)) - \mathbf{e} \rVert^2$. Drop the adjacent-window
  temporal-consistency term until per-axis identifiability is established;
  smoothing a code that carries no identity just yields a smoother wrong signal.
- **Architecture:** keep the causal CNN over a 20-tick (1.0 s) window. It is
  already built, tested and latency-checked.

**Gate 2.** Per-axis $R^2$ of $\text{dec}(\hat z)$ against `e` on held-out
parameter draws, reported per parameter. Leakage and bulk modulus should be the
easiest --- they change the pressure response directly, which the AAM row
observes via `dP`.

### Phase 3 — A-RMA fine-tune

- Unfreeze $\pi$, condition on $\hat z$, normal learning rate and entropy.
- Reintroduce `residual_scale` only as a safety bound, never as a mechanism.

**Gate 3.** Acceptance ordering on the 33-cell panel:
`true z` < `ẑ` < `mean z` < `zero/shuffled`, with `true` vs `mean` resolved.
Report medians and per-cell win counts beside every mean --- E6's whole finding
was that means moved $11$--$25\%$ while medians were flat.

## 8. Relation to the existing `experiments/RMA_PPO/` scaffold

That scaffold is well built and its phase structure matches the above. It
differs in one substantive respect, and the difference is a choice worth making
explicitly.

Its context vector is
`[actuator_authority_alpha, hydraulic_response_time_tau, actuator_delay_steps, mass_scale]`
--- *actuator-stress coordinates*, not hydraulic parameters. That was a sound
way to dodge the "seal is inert on FD" bind, because those coordinates act in
the displacement domain and therefore reach every stack.

Choose deliberately:

- **Stress coordinates** (existing scaffold): reach every stack, work with the
  ID→FD plant unchanged, and give a guaranteed endpoint at $\alpha \to 0$. But
  they are not hydraulic parameters, so the thesis claim becomes "adapts to
  actuator degradation", not "identifies hydraulic plant parameters online".
- **Hydraulic parameters** (this document): the claim you actually want, and it
  requires the §4 repair first.

The two are compatible: `e` can carry both, with the stress coordinates as the
guaranteed-reachable component and the hydraulic parameters as the physically
interesting one. That is the recommended vector if the repair lands.

## 9. Order of work

1. **Repair the ROM physics** (§4). Nothing downstream is meaningful without it.
   Acceptance: physics path carries most of $\Delta x$; no parameter at a clamp.
2. **Establish plant sensitivity** on the repaired plant. Sweep $E_{\text{eff}}$
   over the ch3 range and confirm a resolved task-level effect. This replaces
   the Gate $-1$ of `RMA_ARCHITECTURE_DESIGN.md` and is now much more likely to
   pass, because the earlier nulls were measured on a plant whose physics path
   was switched off.
3. **Phase 1** and Gate 1.
4. **Phase 2** and Gate 2.
5. **Phase 3** and Gate 3.

Steps 1 and 2 are CPU-cheap and are where the risk lives. Step 3 is the
expensive one; time a 20-iteration run before launching it.

## 10. What would falsify the whole line

If, on a repaired plant with live physics and a bulk-modulus range known to move
Simscape's tracking error by $11\%$, a privileged latent still fails to beat a
content-free constant --- then this task is genuinely not plant-limited at these
speeds, and no architecture will make online adaptation pay. That is a clean,
defensible negative result, and step 2 reaches it for hours of CPU rather than
weeks of GPU.
