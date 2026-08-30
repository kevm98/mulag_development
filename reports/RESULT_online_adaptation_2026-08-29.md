# Online adaptation improves task performance under a changing plant

**RMA-style pass.** Pre-registered design, validated instrument, 528 paired
environments across three held-out meshes.

## Result

| arm | cut height | lateral |
|---|---|---|
| frozen base | 7.964 mm | 61.155 mm |
| **adaptive (history live)** | **2.970 mm** | **44.854 mm** |
| control (history zeroed) | 97.105 mm | 252.898 mm |

| comparison (paired median log-ratio) | cut height | 95% CI |
|---|---|---|
| **base vs adaptive** | **+160.8%** | [+134.5, +187.2] |
| control vs adaptive | +1529.4% | [+1170.4, +2108.4] |
| base vs adaptive (lateral) | +35.5% | [+30.5, +41.2] |

**The adaptive policy reduces cut-height error by 63% against the frozen
baseline**, under a plant change the baseline handles badly.

Sharpest statement: the mid-episode mass flip costs the frozen policy **+205.6%**
cut error (CI [+153.6, +268.0]); the adaptive policy scores **2.970 mm**, below
the frozen policy's own UNDISTURBED nominal of 3.550 mm. It more than erases the
degradation.

## Why this is a claim about ONLINE ADAPTATION

The three arms share the frozen base policy, the residual architecture, the
capacity (53,836 trainable parameters), the training budget and the seed. The
residual head is zero-initialised, so every arm STARTS as bit-for-bit the same
policy (verified: max |wrapped - base| = 0.0).

The only difference between adaptive and control is whether the residual can see
the (state, action) history. So the gap is attributable to online sensing, not to
extra parameters.

The plant is built so that sensing is NECESSARY, not merely available:

* the cutter mass **flips between two well-separated modes mid-episode** (1.0x
  <-> 3.0x), so one compromise action profile is wrong in both halves
* the flip instant is **jittered per environment** (20 s +/- 8 s), so it cannot
  be compensated on a timer
* **half the environments flip up and half flip down**, so the direction cannot
  be assumed

Confirmed upstream: the trained residual's action genuinely responds to its
history -- **3.35%** of action scale, against **0.004%** for the privileged
context route, which is why that route's ablation could only return a null.

## Honest caveat

The control arm (97.105 mm) is ~12x WORSE than the frozen base, so it is not
purely "non-adaptive" -- a residual with authority and no sensing signal actively
damages the policy. The +1529% figure therefore mixes "history helps" with "an
unguided residual hurts".

**Lead with the base comparison, +160.8%.** Report the control as evidence that
the latent is necessary for the residual to be usable at all, not as the size of
the adaptation benefit.

## Instrument

Scored through `evaluate_gates.py` with the residual spliced into the actor's
`mlp`, so it inherits `roll_mesh` and every measurement convention in that file.

**Validated before any arm was read:** with `--residual_alpha 0.0` the residual
contributes exactly nothing, and the run reproduced the frozen base at
**7.246 mm** against an independent reference of **7.572 mm** (4%).

A standalone evaluator written earlier was **29x** off the same reference, twice
-- first from a wrong guess about parked steps, then from omitting rsl_rl's
normaliser epsilon (`(x-mean)/(std+1e-2)`; channels with std 8.685e-4 are scaled
12.5x wrong). Every arm number produced through it was withdrawn. Validate a new
instrument against a known value before running anything through it.

## Reproduce

```
# train (both arms, identical but for --no_history)
PPO_AAM_FD/scripts/train_residual_adapter.py <frozen base> \
  --num_envs 256 --iterations 400 --history 20 --latent_dim 8 --alpha 0.5
# plant: MULAG_MASS_BIMODAL=1.0,3.0 MULAG_MASS_STEP_TIME_S=20
#        MULAG_MASS_STEP_JITTER_S=8 MULAG_DIRECT_VALVE=1 MULAG_RMA_CONTEXT=1

# score (three arms x three meshes)
PPO_AAM_FD/scripts/evaluate_gates.py <frozen base> \
  --residual_checkpoint <arm> --residual_alpha {0.0|0.5} [--residual_no_history] \
  --num_envs 176 --steps 900 --meshes {1337|17|4242} --seed 17
```

Pre-registration: [PREREGISTRATION_residual_adaptation.md](PREREGISTRATION_residual_adaptation.md)
