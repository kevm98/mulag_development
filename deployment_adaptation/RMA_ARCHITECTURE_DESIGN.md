# An RMA design for this system, and what it costs to make it honest

Written 2026-08-29. Companion to `AAM_RMA_REPAIR.md`, which gives the gate
order. This document answers a narrower question: *what should the architecture
actually be*, given what the repo has already measured. It changes one of that
plan's assumptions.

---

## The bind, stated first

Three measurements already in this repo, which together rule out the obvious
design:

1. **The latent is a seal latent.** Linear decoding from `z_hat` gives
   seal $0.956$, compliance $0.448$, flow $0.114$, pressure $0.081$,
   com_z $0.056$, mass $-0.008$ (`experiments/E3_mid_episode_plant_change/README.md`).
2. **Seal is inert on the plant AIF runs.**
   `CylinderFDRuntime.step(valve_cmd, extension, extension_velocity, dP, force)`
   takes no factor argument (`mulag_plant/runtime/cylinder_models_runtime.py:320`).
   On the ID→FD path only `mass_scale` and `com_z` vary, through PhysX.
3. **Those are the two factors the latent identifies worst** — $0.056$ and
   $-0.008$, i.e. not at all.

So the factor the estimator reads best cannot be varied where the controller
runs, and the factors that can be varied there are invisible to the estimator.
**E4's oracle null is fully explained by this alone**, with no appeal to
saturation: a privileged latent that encodes seal is worthless on a plant whose
seal does not move. Gate 0's 80% saturation is real and compounds it, but it is
the second problem, not the first.

The design consequence is blunt: **swapping the ID for a direct-command policy
does not fix this.** It removes the saturated anchor and leaves the estimator
still blind to the only factors that vary. That would be an expensive route to
the same null.

## Decision 1 — the plant, not the controller, is the thing to change

Three ways out of the bind. Only one is cheap.

| option | what it does | cost | verdict |
|---|---|---|---|
| **A. Run RMA on the ROM plant** | hydraulic factors act there, so the seal latent becomes informative | small | **recommended** |
| B. Make the FD factor-conditional | rebuild and re-identify the forward model over the factor design | weeks | later, if the thesis needs the ID→FD plant specifically |
| C. Retrain the latent onto mass/com_z | keep the ID→FD plant, re-encode `z` onto the two factors that act | medium | weak: those factors are narrow, and `com_z` is unidentifiable without Gate 2 |

**Option A is also already RMA-shaped, which is easy to miss.** The R stack is

```
pi(x) -> desired cylinder motion -> command ROM -> PhysX
```

The command ROM is the *plant*, not an inverse model. There is no frozen
inverse model in that path and therefore nothing to saturate: Gate 0's 80% is a
property of the ID→FD path and does not exist here. Textbook RMA is
`pi(x, z) -> action -> plant`, and R differs from it in exactly one respect —
the policy is not conditioned on `z`.

So the architecture change is not a rebuild. It is: **add two observation
columns to the ROM policy and train it with the privileged latent.**

## The recommended architecture

```
                 e (6 privileged factors, episode-constant)
                 |
                 v
        mu(e) -> z (d_z = 2)          <- trained WITH the policy under RL,
                 |                        not with SD_z
                 v
  x ----> pi(x, z) --> desired cylinder motion --> command ROM --> PhysX
                 ^
                 |
        z_hat = phi(history)          <- phase 2; regressed onto the SAME mu
                                         that phase 1 produced
```

Phases follow Kumar et al. exactly, which is what makes this defensible as a
method rather than as a local invention:

- **Phase 1.** Train `pi(x, z)` with privileged `z = mu(e)`, `mu` learned jointly
  under the PPO objective. Auxiliary loss `||dec(z) - e||^2` so the two latent
  axes acquire physical meaning; without it `d_z = 2` stays an unstructured
  bottleneck and phase 3's per-axis checks have nothing to check.
- **Phase 2.** Freeze `pi` and `mu`. Train `phi(history) -> z_hat` by regression
  onto `z`, on **closed-loop rollouts of the policy that will deploy** — not on
  the existing 2.3 GB PRBS identification collection, which is
  classical-controller excitation and the wrong distribution.
- **Phase 3 (A-RMA).** Fine-tune `pi` on `z_hat`. Normal learning rate and
  entropy; obs-normaliser refit including the `z` columns.

Keep `d_z = 2`. It matches the AAM's existing output shape and the +2
observation layout the AIF env already uses, so `phi` and the obs plumbing carry
over unchanged. Revisit only if phase-1 per-axis reconstruction shows the
bottleneck binding.

## Decision 2 — the prerequisite nobody has run

**This is the addition to `AAM_RMA_REPAIR.md`, and it outranks Gate 1.**

Adaptation can only pay where plant variation measurably changes task
performance. On this task, at the operating point everything has been measured
at, it does not:

- E3: a 40% loss of valve effectiveness, delivered exactly as configured mid
  episode, moved post-change lateral error from $98.1$ mm to $95.3$ mm — $-2.9\%$,
  the wrong direction. "The stack absorbs it, so there is nothing to recover
  from."
- E5: the ROM authority ladder shows no contact degradation until $\alpha \le
  0.03$, i.e. until $97\%$ of actuator authority is gone.
- The ROM's direction guard removes $79.5\%$ of a reference plant change
  ($93.16$ mm $\to$ $19.07$ mm) before it reaches the task.

A z-conditional policy trained in that regime will correctly learn to ignore
`z`, because on the evidence `z` does not predict anything about the return.
Gate 1 would then produce a null that says nothing about RMA.

**Gate $-1$: establish a regime where the plant matters.** Sweep the operating
point until a *privileged* contrast — nominal plant against a held-out
combined_test draw, same policy, same seeds — produces a resolved gap in lateral
error or contact. Candidate knobs, cheapest first:

1. **Speed.** Everything above was measured at 4 km/h; the trained range is
   2--7. Less time to correct should make the plant matter more.
2. **The direction guard.** It is the largest single absorber. Measure with it
   at the deployment setting and off.
3. **Trajectory family and mesh.** E5 already shows an 11.0 pp contact spread
   across meshes at nominal.
4. **Factor range.** Last resort: beyond the identified envelope the ROM is
   extrapolating, and the result stops being about the real machine.

Pass criterion: a plant-only contrast with a bootstrap 95% CI excluding zero.
Until that exists, **no RMA phase is worth running**, and that itself is a
publishable finding — "this task is not plant-sensitive in the regime studied"
is a real result and cheaper to establish than to work around.

## Decision 3 — what "online" buys, and the honest version of the claim

`e` is drawn once and held for the whole run
(`deployment_adaptation/README.md:108`), and episodes are $47.3$ s. This is the
standard RMA setting — Kumar et al. also sample environment parameters per
episode — but it fixes what the adaptation is *for*: **fast identification at
episode start**, not tracking a drifting plant.

Two consequences to design around rather than discover later:

- The window in which `z_hat` differs from a converged constant is the
  identification transient, on the order of the AAM's 1.0 s history against a
  47.3 s episode. If the metric averages over the whole episode, a real
  adaptation benefit is diluted roughly 50:1. **Score the transient separately**,
  or shorten episodes for the adaptation panel.
- Any claim of *tracking a time-varying plant* needs a plant that varies within
  the episode. E3's mid-episode step is the mechanism; its v2 replacement is
  specified and unrun. Keep the two claims apart: episode-constant `e` supports
  "identifies the plant quickly", not "tracks a changing one".

## Decision 4 — safety, once the ID is not in the path

On the ROM path this is less acute than under Option A-as-originally-framed,
because no frozen inverse model was ever acting as the anchor. Still required:

- the eight acceptance gates unchanged, and `c3_physical_extension_reserve_m`
  kept at its current value, so the physical contract matches the existing
  scored runs;
- action rate limiting retained, since a `z`-conditioned policy is a new policy
  and its failure modes are not the old one's;
- the `actuator_stress.py` chain left at nominal for every RMA run, so the
  adaptation result is not entangled with the E5 stress axis.

## Validation, and what would falsify this

Every phase gets a positive control, because on this task a null is the default
outcome and must be distinguishable from a broken instrument.

| phase | gate | falsifier |
|---|---|---|
| $-1$ | privileged plant contrast resolves | no regime found $\Rightarrow$ task not plant-sensitive; stop and report |
| 1 | `zero`/`shuffle` ablation hurts a `z`-trained policy | no gap $\Rightarrow$ `z` carries nothing the policy can use |
| 2 | per-axis $R^2$ of `dec(z_hat)` against `e`, held-out draws | seal recoverable but com_z not $\Rightarrow$ Gate 2 excitation first |
| 3 | acceptance ordering `true z < z_hat < mean z < zero/shuffled` | `z_hat` above `mean z` $\Rightarrow$ estimator problem, return to phase 2 |

Report medians and per-cell win counts beside every mean. E6's entire finding
was that means improved $11$--$25\%$ while medians were flat; a mean-only table
would have read as success.

## What this costs

Gate $-1$ is a scoring sweep, no training: hours, single-digit dollars. Phase 1
is the expensive item — a from-scratch PPO run on an L4, and the E5 ladder took
$2$ h $22$ m for 48 scored runs, so a training run is a different order.
**Time a 20-iteration run and bring the number before launching the full one.**

## The one-line summary

The ID is not the first problem. The first problem is that the estimator reads a
factor the deployment plant does not vary, and the second is that the task
barely responds to plant variation at all. Moving RMA onto the ROM plant fixes
the first for the price of two observation columns. The second has to be
measured before anything is trained, and it may end the line — which is itself
the honest result.
