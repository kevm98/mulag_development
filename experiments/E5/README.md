# E5 — Plant robustness envelope

**Defends:** *Hydraulic Manipulator* · **Covers:** all three stacks
**Status:** superseded sweep CLOSED (thesis §7.4); replacement SPECIFIED, not run

## What the first sweep did, and why its result was withdrawn

`code/run_e5_robustness.sh` scaled the whole six-factor draw at 0, 0.5, 1, 1.5
and 2x and found a flat curve on every stack. That was reported as "no stack
degrades with the size of the plant deviation". **That conclusion is not
supported by the experiment**, for two reasons found afterwards. Both are
checkable without a GPU.

### 1. The factors do not all reach the plant

`python AAM_module/tools/audit_factor_clamping.py` and inspection of the
runtime:

| factor | R (command ROM) | IF / AIF (ID→FD) |
|---|---|---|
| flow, compliance, seal, pressure | acts, **clamped** to the identified range | **inert** — `CylinderFD.step()` takes no factor argument |
| mass_scale, com_z | acts, unclamped (PhysX) | acts, unclamped (PhysX) |

`RomFactorPerturbation` drives `_command_rom_system_factors`, read only in the
`command_rom` branch of `mulag_upper_level_env`. IF and AIF set
`cylinder_dynamics_model = "fd"`. So for two of the three stacks the sweep
varied tool mass and centre-of-mass height and nothing else, and a flat curve
was close to guaranteed.

### 2. What does act is clamped, and bounded

`HydraulicROM.forward` pins compliance to [-0.25, 0.10], flow to [-0.25, 0.25],
pressure to [-0.15, 0.15] and seal to [0, 1], then clamps the response gain to
[0.35, 1.50]. Over the 16 held-out rows:

| requested scale | rows with a pinned factor | delivered hydraulic deviation |
|---|---|---|
| 1.0 | 0/16 | 1.000 |
| 1.5 | 13/16 | 0.911 |
| 2.0 | 16/16 | 0.778 |
| 3.0 | 16/16 | 0.586 |

**At most 62% of actuator authority is removable through this path**, so a
zero-authority knockout is unreachable and a breakdown point cannot be found by
scaling factors. Extending the sweep to scale 3 delivers *less* of what it
requests, not more. This also explains the non-monotonic IF excursion: scale 1
is the last point where the manifest draw arrives intact, and above it the
hydraulic half stops moving while the mechanical half keeps growing, so the axis
changes the *kind* of plant rather than the amount of deviation.

### What "62%" does and does not mean

Two different quantities, kept apart:

- **delivered fraction** (0.586 at scale 3) — the norm of the *applied* hydraulic
  deviation over the norm of the *requested* one. A ratio in factor space. It
  says how much stress survived clamping, **not** what it does to the machine.
- **response gain** — multiplies both the velocity and the extension increment
  the ROM emits, so it scales realised cylinder motion directly. The explicit
  limiter clamps it to [0.35, 1.50], permitting at most a 65% reduction; the
  worst case the six clamped factors can actually compose is **0.376**, i.e. at
  most **62% of the ROM's realised actuator motion** is removable. On the ID→FD
  plant the same factors scale nothing at all.

## The replacement

`code/check_factor_reachability.sh` — Stage 0a. Scores each stack under factor
draws differing only in the hydraulic channels. Identical reports confirm
inertness empirically rather than by reading the source.

### The stress axis

`actuator_authority_alpha` scales the realised cylinder motion the actuator
produced, immediately before it updates the PhysX target:

    R      : Δx from the command ROM → α·Δx → target
    IF/AIF : Δx from ID→FD          → α·Δx → target

Same location, same meaning, all three stacks. It bypasses the ROM clamp and
reaches the FD plant, which no factor does. It is **actuator-authority
degradation at the deployment interface** — not seal, not flow, not a hydraulic
parameter — and it has an obvious endpoint: at α→0 the reference must become
untrackable.

`code/run_e5_stress_envelope.sh` — four stages, cheapest first:

- `STAGE_E5=0` **authority knockout**, all three stacks, ~6 episodes. α=1.0 vs
  α=0.01, with `check_authority_applied.py` verifying |Δx_applied|/|Δx_raw| = α
  **numerically** from the trace before anyone looks at the cutter. Two distinct
  failures are separable: factor not applied (ratio ≠ α) vs applied and still
  tracking (ratio = α, error flat) — the second is a harness fault further
  along.
- `STAGE_E5=C` **clamp sentinel**, R only, 2 episodes. `flow=-0.40` and
  `flow=-0.80` both clamp to −0.25 and must agree to numerical tolerance, with
  mass and COM nominal in both (the inline factor spec leaves unnamed factors at
  nominal, so nothing else varies). `flow=-0.10` is the control for the control:
  inside the range, so it must differ.
- `STAGE_E5=A` **degradation discovery**, **IF only**, 3 families, ~17 episodes.
  α over {1.0 … 0.01} plus a native mass scan. `summarise_authority.py` applies
  the predeclared thresholds, locates the knee and **freezes** the levels to
  `results/stress_levels.json`.
- `STAGE_E5=B` **frozen-level evaluation**, all three stacks, 11 families.
  Refuses to run without the frozen levels and does not re-select.

**Selection discipline:** the curve is measured on IF alone, the levels are
frozen, and only then are R/IF/AIF scored. AIF is never inspected while the
levels are chosen, so the condition cannot have been picked to flatter it.
`--write-levels` is rejected unless `--stack if`.

Factors can be placed at a stated point with `FACTOR_KEY="mass_scale=1.75"` or
`FACTOR_KEY="flow=-0.4,seal=1.0"` (added to `verge_task/deployment_factors.py`),
deliberately including values outside the audited ranges. The ROM's own clamp
still applies to hydraulic channels — that is what the clamp control measures.

## Thresholds, declared before the curve is seen

- **Degradation onset** — first stress at which median lateral error reaches
  1.5x its own nominal value, or ground contact falls by ≥5 percentage points.
- **Functional breakdown** — ground contact < 0.700, or safety termination rate
  > 0.080.

The figures this produces: α → lateral error, α → ground-contact fraction (with
the 0.700 line), and α → error relative to nominal, each with R, IF and AIF.

## Traps

- This project has repeatedly produced **vacuous gates** — thresholds no run
  could violate. Every gate needs a measured floor and a positive control.
- A flat curve has two readings: the controller absorbs the stress, or the
  stress never arrived. Stage 0 exists to tell them apart.
- Sequential audits in one process have fabricated failures; reproduce any
  failure in a fresh process before believing it.

## Stage 0b — effective bulk modulus as a stress axis (CPU, run)

`code/rom_bulk_modulus_sweep.py` → `results/rom_bulk_modulus_sweep.json`

Runs E5's own 0 / 0.5 / 1 / 1.5 / 2 grid on the effective bulk modulus beta
alone, through the deployed bridge `CommandCylinderROMRuntime` — the same class
`mulag_upper_level_env` uses in its `command_rom` branch. CPU only, no Isaac.

**The grid is written into `log_bulk_modulus` directly, not through
`system_factors`.** The compliance channel pins e_comp to [-0.25, 0.10], so it
can only deliver 0.779x..1.105x of nominal beta — it cannot express 0x or 2x at
all. That is the same clamp that hollowed out the superseded sweep above.
Delivered fraction is 1.000 at every point; the only point the model's own
`[0.01, 3.0]` GPa clamp touches is beta=0, which becomes 0.01 GPa — a 19.6x
softening, not a true zero.

Nominal learned beta is per-cylinder [0.231, 0.280, 0.149, 0.121] GPa, so the
grid read as absolute GPa is a *stiffening* sweep; both readings are reported.

### The probe trajectory decides the answer — read this before the numbers

`command_rom_direction_guard` sets its limit proportional to `|command|`, so an
axis sitting at zero command has its learned response forced to exactly zero.
The runtime's own source flags this (`mulag_upper_level_env.py`): a single-axis
probe reaches 0.073 of its travel against 0.827 for direct drive. A trajectory
that drives one cylinder at a time therefore holds 3 of 4 axes in that dead
state and **suppresses the measured sensitivity to every hydraulic parameter**.

Both probes are run. `all_axes` — phase-shifted sinusoids, every cylinder
commanded every tick — is the deployment-like one and is the result. `staggered`
is kept only to quantify the artifact.

| probe | guard veto | authority kept | residual share of Δx | beta=0 ext err |
|---|---|---|---|---|
| all_axes | 15.9% | 72.5% | 31.1% | **51.50 mm** |
| staggered | 87.5% | 27.4% | 70.9% | 0.42 mm |

Same model, same grid, same guard: a 122x difference in apparent degradation.
**Any hydraulic-factor null measured on a single-axis probe is uninterpretable.**

### Result (all_axes, guard on — the deployment configuration)

80 ticks, nominal travel 37.60 mm:

| beta | delivered | rel dP | rel vel | rel ext | peak ext err |
|---|---|---|---|---|---|
| 0x (→0.01 GPa) | 1.000 | 528.3% | 64.6% | **148.3%** | 51.50 mm |
| 0.5x | 1.000 | 88.5% | 39.2% | 54.8% | 19.02 mm |
| 1x | — | — | — | — | 0 |
| 1.5x | 1.000 | 28.9% | 16.4% | 24.5% | 8.49 mm |
| 2x | 1.000 | 42.8% | 23.8% | 39.7% | 13.78 mm |

**Effective bulk modulus is a live degradation axis.** It is fully delivered, it
acts on the hydraulics, and it reaches the plant: softening to 0.5x already
displaces the realised extension by 19 mm on 37.6 mm of nominal travel, and
collapsing beta exceeds the nominal signal range outright.

Softening degrades far more than stiffening (0.5x → 19.0 mm vs 1.5x → 8.5 mm),
consistent with pressure rate ∝ beta. Stiffening saturates hard: 0.5 → 2.0 GPa
absolute, a 4x change, moves the error only 17.4 → 20.4 mm.

### Where the absorbers still sit

Guard-on and guard-off differ, so both are reported. Under `all_axes` the guard
still passes 72.5% of raw displacement and vetoes only 15.9% of channel-ticks,
and the beta-independent residual heads supply 31.1% of the extension increment
(against 70.9% on the suppressed probe). The physics path dominates here, which
is why beta bites.

### Verdict

Effective bulk modulus is a **usable** stress axis for the ROM stack — delivered,
monotone in |log beta| on each side, with a working positive control. It is not
a substitute for `actuator_authority_alpha`, which still has the guaranteed
alpha→0 endpoint this axis lacks (the `[0.01, 3.0]` GPa clamp bounds beta from
below), but the earlier claim that hydraulic factors cannot reach this plant does
not survive an all-axes probe.

**Scope.** Open-loop probe, perfect-follower plant surrogate, two trajectories,
one checkpoint (`plant_reachable_v1_finetune_blend25`). It bounds the open-loop
effect; a closed PPO loop could amplify or absorb it.
