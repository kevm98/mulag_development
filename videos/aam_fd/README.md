# PPO-AAM-FD — online adaptation under a mid-episode mass flip

Rollouts on mesh 1337, recorded 2026-08-29/30 on the GCP L4. These are the
visual companion to `reports/RESULT_online_adaptation_2026-08-29.md`.

The plant flips the cutter mass between two well-separated modes (1.0x <-> 3.0x)
partway through the episode. The instant is jittered per environment
(20 s +/- 8 s) and half the environments flip up while half flip down, so the
change can be neither timed nor assumed.

All three scored arms share the same frozen base policy, the same residual
architecture, the same 53,836 trainable parameters, the same budget and the same
seed. The residual head is zero-initialised, so every arm starts bit-for-bit
identical. The only difference between `adaptive` and `control` is whether the
residual can see the (state, action) history.

| file | arm | cut height | what to watch for |
|---|---|---|---|
| `nominal_mesh1337.mp4` | undisturbed reference | 3.550 mm | no flip occurs |
| `base_mesh1337.mp4` | frozen policy, flip active | 7.964 mm | degrades at the flip, +205.6% cut error |
| `adaptive_mesh1337.mp4` | history live | **2.970 mm** | recovers, and lands below the undisturbed nominal |
| `control_mesh1337.mp4` | history zeroed | 97.105 mm | residual has authority but no sensing, and actively harms |

Watch `base` then `adaptive` back to back: same policy underneath, same
disturbance, and the only difference is whether the residual can see its own
history.

Read `control` as evidence that the latent is necessary for the residual to be
usable at all, not as the size of the adaptation benefit -- it is ~12x worse
than the frozen base, so the headline number is the base comparison, +160.8%
(95% CI [+134.5, +187.2]).
