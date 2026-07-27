# Circle c7 independent audit and root cause

Date: 2026-07-26

## Decision

Circle c7 is **not certified**. The seed-29 audit is protocol-valid (32/32
rollouts, fresh process per rollout, correct model and c7 hashes, factor-blind,
corrected clearance), but Monte Carlo fails the 40 m/s2 p95 acceleration gate.

- MC seed 300037: 40.8448 m/s2
- MC seed 300042: 41.7117 m/s2
- every correlated h condition: pass
- Monte Carlo and h termination confidence: pass
- tracking, alignment, clearance and truck point gates: pass

## Measured mechanism

1. Acceleration is primarily circle-phase/configuration dependent. The failing
   profiles still peak around bins 2-4; seed 300037 also has a bin-0 tail and
   seed 300042 has smaller bin-6 activity.
2. No individual ROM factor explains the tail. Across the two failures, the
   absolute Pearson association between each factor and per-environment p95
   acceleration is at most 0.09. All fixed correlated h plants pass.
3. PPO residual action supplies the final gate violation. On matched fresh
   zero-residual ablations:
   - seed 300037: 40.8448 -> 39.69 m/s2;
   - seed 300042: 41.7117 -> 39.57 m/s2.
4. The baseline reference/inner-loop/plant response is nevertheless already
   only 0.3-0.4 m/s2 below the gate. PPO is therefore not the sole physical
   source; it turns an insufficient-margin baseline into a failure.

The root cause is not simulator leakage, clearance measurement, truck motion,
or a single hidden factor. It is a configuration-dependent closed-loop
acceleration tail with inadequate baseline margin, amplified by the existing
PPO residual on some independently randomized plant/start-phase combinations.

Further timing changes based on these seeds would repeat the overfitting pattern.
The next justified remediation is a small circle-focused PPO fine-tune that keeps
the c7 reference and existing acceleration reward, followed by a new independent
seed family. ROM-C should not be used to hide this ROM-B baseline defect.

## Superseding zero-baseline result

The later full seed-41 zero-residual sweep supersedes the policy-first
recommendation above. It found 20/32 rows above 38 m/s2 and 6/32 above the formal
40 m/s2 gate with PPO identically zero. PPO training is therefore paused. The
baseline reference/inner-loop/ROM/PhysX response must be investigated first; see
Section 14 of `ROM_B_CIRCLE_COMPLETE_TECHNICAL_HANDOFF_2026-07-26.md`.
