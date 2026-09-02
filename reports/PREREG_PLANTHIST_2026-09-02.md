# Pre-registration — plant-only history (the encoder-overfitting hypothesis)

Fixed **2026-09-02 01:20 UTC** (03:20 CEST), before any arm of this experiment
exists and before any of them is scored. The fresh test set named below has
never been evaluated against by any run in this repository.

## What prompted this

The triplet experiment (`PREREG_TRIPLET_2026-09-01.md`) returned a negative
primary result and triggered its own falsification condition:

| axis | result |
|---|---|
| control -> adaptive (live history vs matched capacity) | 1 of 5 seeds succeeds; median FAILS |
| base -> control (capacity alone) | +65.1%, +76.5%, +81.4%, +1.5%, +85.1% on cut |

The capacity-matched control does not merely match the adaptive arm; on cut
height it **beats** it on three of five seeds (-27.0%, -19.7%, -47.4%).

**A network fed real data should not systematically lose to the same network fed
zeros.** That is not the shape of a null result, and this experiment tests the
specific mechanism that explains it.

## The hypothesis

Every arm ever trained in this repository — R1, R2, R3 and the triplet — passed
`--history 20` **without** `--plant_history`, so the encoder received the full
observation: `66 + 4 = 70` features per step, **1400 numbers** over the window,
almost all terrain and task state rather than plant identity.

`verge_task/residual_adapter.py:76` already records what that does:

> "A 53k-parameter head has ample room to fit that noise, which is why the
> benefit swung +1529% to -80% across seeds in BOTH directions."

That is a description of the observed table. The hypothesis is therefore:

> **The seed variance and the adaptive-loses-to-control inversion are caused by
> the history encoder overfitting terrain noise, not by online adaptation being
> useless on this plant.**

Under this hypothesis the zeroed-history control is stable and strong precisely
*because* zeros inject no noise, which is what the data shows.

Two independent results in the literature predict the same failure:

* **RMA (Kumar et al. 2021)** never trains the history encoder with RL. Phase 2
  fits it by supervised regression `MSE(z_hat, z)` onto a low-dimensional
  privileged extrinsics vector, specifically to avoid a high-capacity encoder on
  high-dimensional history under a noisy RL gradient. Their ablation: 73.5% with
  adaptation, 52.1% without.
* **RESeL (arXiv 2405.15384)** shows context encoders require a *lower* learning
  rate than the policy MLP or training destabilises. This also explains why R2
  failed: a global drop to 1e-4 under-trained the policy without treating the
  encoder, so the uniform learning rate was the wrong knob.

## The single change

```
--plant_history
```

The encoder input becomes the plant signature only — cylinder extensions 12:16,
velocities 16:20, plus the commanded action: `8 + 4 = 12` features per step,
**240 numbers** instead of 1400. This is what was COMMANDED against how the
cylinders RESPONDED, and it is invariant to where the machine sits on the
terrain, which is the property a plant latent needs.

Everything else is identical to the triplet arms: same frozen base, `--num_envs
256 --iterations 400 --history 20 --latent_dim 8 --alpha 0.5 --lr 3e-4`,
`--lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0`, same plant change,
same five seeds, same triplet structure {base, zero-history control, live-history
adaptive}, same uniform best-training-reward checkpoint rule applied by the same
parser to every arm.

One change, so if the result differs the cause is attributable.

## Fresh test set — meshes 2718, 3141, 8191

The previous meshes (1337, 17, 4242, 909, 7001) have now been looked at five
times. A nominal 95% interval is not a 95% interval across five attempts, and no
caveat repairs that.

Meshes are arbitrary procedural terrain seeds
(`PPO_ROM/scripts/evaluate_gates.py:736`). **2718, 3141 and 8191 have never been
evaluated against by any run in this repository** — verified against every report
in `outputs/rif_delay/`, which contains only `_m{17,55,909,1337,4242,7001}`.

This is a clean test set and it resets the multiplicity problem rather than
apologising for it. The base arm is re-scored on these meshes; nothing is
carried over.

## Primary prediction — falsifiable, quantitative, fixed here

The claim is about VARIANCE, not just the mean. On the control -> adaptive axis
under the 1400-dim history, the cut-height effect spans

    -47.4% .. +30.9%   =   range 78.3 percentage points

across the five seeds.

> **PREDICTION: with plant-only history the across-seed range of the
> control -> adaptive cut effect is less than half that, i.e. under 40
> percentage points.**

**If the range stays wide, the encoder-overfitting hypothesis is REFUTED** and
that is the reported result, regardless of what the mean does. A method that
merely shifts the mean while staying this seed-unstable has not been fixed.

Secondary prediction: no seed shows the adaptive arm significantly WORSE than
its own capacity-matched control on both metrics.

## Success criterion — unchanged

Identical to `PREREG_TRIPLET_2026-09-01.md`, not re-decided here: an arm
succeeds if one metric improves significantly (paired bootstrap CI excluding
zero) while the other improves or stays within its measured non-inferiority
bound (cut 10%, lateral 1.5%, the A/A floors from `PREREG_R2_2026-09-01.md`).
The claim holds only if the median of the five seeds succeeds.

## Reporting

All five triplets are reported with their outcome, whatever it is. The negative
triplet result stands as reported and is **not** superseded or withdrawn by this
experiment: it is the result for full-observation history, and it is what
motivated this test. If this experiment also fails, both negatives are reported
together with the diagnosis, and the honest conclusion is that online adaptation
does not beat matched capacity on this plant.
