# Model Fine-Tuning Live Status

**Last updated**: 2026-07-01

---

## FD Fine-Tuning

| Metric | Value |
|--------|-------|
| Dataset samples | 62,400 (12 episodes x multi-cylinder sweeps) |
| Extension range | [0.328, 0.631] — 0.3m range ✅ |
| Valve-delta per-step correlation | ~0.0 (implicit actuator oscillation) |
| Original FD sign accuracy (on dataset) | **0.88** ✅ |
| Fine-tuned FD sign accuracy | 0.54 ❌ |

**Decision**: Original FD is ALREADY GOOD (88% sign accuracy with temporal history). Fine-tuning on the lower-quality PhysX data degrades performance. Keep original FD.

---

## ID Fine-Tuning

| Metric | Value |
|--------|-------|
| Teacher data | Valve-space PID (kp=40, ki=15, kd=2) |
| Good samples (all 4 cyls sign-correct) | 1,369 / 9,600 (14%) |
| ID model | VelocityToValveMLP (16->128->128->64->4) |
| Validation sign accuracy | **97.2%** ✅ |
| Validation MAE | 0.086 |

**Fine-tuned ID learns velocity→valve mapping with 97% sign accuracy on held-out data.**

---

## Full Pipeline Validation (fine-tuned ID + original FD + adapter + PhysX)

| Cyl | Best Sign Acc | Best Attenuation | Notes |
|-----|--------------|-----------------|-------|
| 0 | 0.66 | 0.79 | Oscillatory, valve bias negative |
| 1 | 0.82 | 1.16 | Best positive tracking, gravity limits negative |
| 2 | 0.92 | 0.57 | Excellent negative sign at -0.02 |
| 3 | 0.76 | 1.15 | Valve bias positive, delta tiny |

**Verdict**: Pipeline PARTIALLY WORKS. Fine-tuned ID produces correct-signed valve commands (inheriting the PID teacher's limitations). Original FD produces correct-signed deltas (88% sign acc in isolation). But the combined chain still suffers from:
1. Implicit actuator oscillation (velocity sign flips at control rate)
2. FD signal attenuation (delta magnitudes 0.0001-0.0006m, too small)
3. PID teacher only 14% good → ID learns imperfect mapping

---

## Root Cause

The valve-space PID cannot reliably track cylinder velocity with implicit position actuators. The PID produces valve commands, FD converts to position deltas, but the per-step delta is ~0.0001-0.0006m — too small to overcome gravity/oscillation. The PID teacher data reflects this limitation, and ID inherits it.

The fundamental physics: implicit actuators are POSITION drives, not VELOCITY drives. The Werner paper's controller works because real hydraulics use FLOW control (valve directly controls oil flow = cylinder velocity). Our simulation uses position drives with stiffness/damping, which can't replicate flow control.

---

## Next Steps

1. Use fine-tuned ID (97% sign acc) as feedforward in `id_ff_plus_cylinder_velocity_pid` mode
2. Add PID residual for improved tracking
3. Restore upper-level PPO on this pipeline
4. OR: Accept position-ramp PI as a practical substitute for the learned pipeline
