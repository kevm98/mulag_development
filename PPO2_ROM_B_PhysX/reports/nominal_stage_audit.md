# ROM-B nominal-stage feasibility audit

## Decision

ROM-A / ROM-B Stages 1-3 are prevalidated. Robust PPO training must start at
Stage 4 (`randomization_scale=1.0`) from `model_607.pt`; it must not relearn
behavior already accepted at nominal, mild, or medium randomization.

## Contract corrections

- Promotion uses the repository-standard 120 scored steps per shape.
- At least 95% of replicas must complete (`termination_rate <= 0.05`).
- Tracking, acceleration, head, clearance, and drift are reduced over completed
  replicas. Every failed or incomplete replica remains in the termination rate.
- Five shapes use fixed held-out seeds across curriculum rounds.
- The accepted action authority remains 2 mm for cylinders 1-3 and 0.25 mm for
  cylinder 4. Doubling authority degraded clearance and was rejected.

## Three-run nominal result

The following full audits used 128 environments, all five shapes, 120 scored
steps per shape, nominal factors, deterministic actor-mean actions, and seed 17:

- `logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-02-18/rom_b_curriculum_preflight.json`
- `logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-06-42/rom_b_curriculum_preflight.json`
- `logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-10-50/rom_b_curriculum_preflight.json`

All three runs were bit-identical and passed:

```text
max shape mean tracking error       0.034535 m   <= 0.040
max shape p95 tracking error        0.062508 m   <= 0.080
max shape p95 tool acceleration    39.258484     <= 40.0 m/s^2
minimum head alignment              0.980266     >= 0.980
maximum completed clearance error   0.195661 m   <= 0.200
maximum post-constraint drift       0.00099998 m <= 0.001
maximum termination rate            0.03125      <= 0.05
```

## Feasibility comparisons

- Zero residual did not pass: sine clearance remained unsafe and arc/square
  acceleration exceeded the gate.
- Maximum 4 mm / 0.5 mm authority did not pass and worsened clearance.
- At accepted authority, observed failures were hydraulic tracking/clearance or
  occasional head events; no cylinder-limit, post-constraint truck, or
  non-finite failures dominated the audit.
- A 240-step stress screen was retained as a later robustness evaluation but is
  not the curriculum promotion horizon used by the accepted PPO contract.

## Randomized-stage boundary

Full 128-environment, five-shape audits located the first unsupported stage:

```text
Stage 2, c=0.4: PASS, max termination rate 0.0390625
Stage 3, c=0.7: PASS, max termination rate 0.0234375
Stage 4, c=1.0: FAIL, line termination rate 0.0625
```

All Stage-4 completed-replica quality metrics passed; only line completion was
below 95%. Evidence:

- `logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-16-15/rom_b_curriculum_preflight.json`
- `logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-20-29/rom_b_curriculum_preflight.json`
- `logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-24-44/rom_b_curriculum_preflight.json`

The production trainer starts at Stage 4, allocates three times the iterations
to failed shapes, and rolls back actor, critic, and optimizer whenever a
candidate gate score regresses.
