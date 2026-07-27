# ROM-B circle retiming report

Date: 2026-07-26

## Outcome

The accepted working candidate is `plant_reachable_trajectory_library_v3_circle_c4_retime.json`.
It preserves every circle tool point and every cylinder command from c3. It adds
only a smooth timing coordinate, slowing locally by at most 18.6% around the
measured difficult circle region and recovering nominal speed elsewhere.

Status: **adversarially screened; full circle-only audit pending**. This report
does not claim certification.

## Measurement correction discovered first

The previous isolated evaluator used one process per `(suite, h, shape)` but ran
all seed groups sequentially inside that process. This still leaks simulator
state. For the same `h=1.0`, circle, seed 400030 rollout:

- after rounds 1 and 2 in the same process: 44.0805 m/s2;
- first rollout in a fresh process: 39.99 m/s2.

The evaluator now launches one fresh process per rollout seed. Shape-subset
runs also construct on canonical `line`, matching the full audit before switching
to the requested shape. New reports record both the trajectory library path and
SHA-256. The old `evaluation_isolated` report is therefore not valid under the
new per-rollout isolation requirement.

## Genuine remaining baseline failure

After removing inherited-state inflation, the discriminating real failure was:

| candidate | h | seed | mean error | p95 error | p95 acceleration | clearance | termination |
|---|---:|---:|---:|---:|---:|---:|---:|
| c3 baseline | 0.4 | 400030 | 0.02610 m | 0.04925 m | 40.98 m/s2 | 0.11795 m | 1/128 |
| c4 retimed | 0.4 | 400030 | 0.02470 m | 0.04526 m | 37.88 m/s2 | 0.09955 m | 0/128 |

The matched diagnostic localized the strongest acceleration occupancy to
geometric path `s=1.04..1.38 m` (bin p95 57.40 m/s2). In that interval, the c3
cylinder path has p95 `|dq/ds|=0.1976` and `|d2q/ds2|=1.9004`.

## Retiming method

The runtime now keeps two separate monotone coordinates:

- geometric `s`: exact circle phase, start sampling, diagnostics and reporting;
- timing `u`: progression clock used to interpolate the unchanged path.

Sampling is uniform in geometric `s` and then mapped to `u`. This is essential:
the first prototype sampled uniformly in stretched `u`, over-populated the hard
region, and produced an invalid baseline/candidate comparison.

For measured bins above the 38 m/s2 development target, the local first
candidate is `speed_scale = sqrt(target / measured)`, the fixed-path curvature
relationship used only to propose a candidate. Gaussian shoulders make the
speed law smooth. PhysX measurement, not the formula, decides acceptance.

c4 has geometric length 3.45565 m, timing length 3.64608 meter-equivalent units,
and minimum local speed scale 0.81361. Cartesian geometry, cylinder positions,
PPO checkpoint, inner controller, ROM plant, and deterministic truck are unchanged.

This follows the path/time separation in TOPP-RA
(https://arxiv.org/abs/1707.07239), with the same motivation for acceleration
and jerk continuity described by TOPP3 (https://arxiv.org/abs/1609.05307) and
Ruckig (https://arxiv.org/abs/2105.04830). No actuator limit was invented: the
schedule was proposed from measured path-bin acceleration and accepted only by
fresh-process PhysX rollouts.

## Adversarial c4 screen

| condition | p95 acceleration | mean / p95 tracking | clearance | termination |
|---|---:|---:|---:|---:|
| h=0.4 round 3 | 37.88 | 0.02470 / 0.04526 m | 0.09955 m | 0/128 |
| h=1.0 round 3 | 38.20 | 0.02472 / 0.04520 m | 0.09103 m | 0/128 |
| h=1.1 round 2 | 36.76 | 0.02474 / 0.04577 m | 0.10724 m | 1/128 |
| Monte Carlo round 2 | 35.95 | 0.02399 / 0.04482 m | 0.09739 m | 0/128 |
| Monte Carlo round 3 | 35.98 | 0.02422 / 0.04428 m | 0.09914 m | 0/128 |

All formal point gates passed. Worst acceleration was 38.20 m/s2. The stricter
38.0 development margin was missed by 0.20, so c4 is not described as having a
full 5% acceleration margin.

One stronger schedule was tested once and rejected: it regressed h=1.0 to
39.75 m/s2 and clearance to 0.19131 m. It was deleted; no parameter search was
continued. A two-update circle PPO pilot also regressed its internal candidate
gate and the training code rolled it back automatically. No new PPO checkpoint
was accepted.

## Reproducibility

- c3 SHA-256: `f84a6ce25ba419529bb8aca7d5947f5c7e8065638bc6525b632fb3baf7e2c4e3`
- c4 SHA-256: `0ea7bc8f0c58f3b315f51a99cc5f9ff61f1008bdc0e8c0ad6819ee33815df94e`
- Tests: 75 passed.

Generate c4:

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_PhysX/scripts/build_circle_retiming_candidate.py
```

Final circle-only audit (5 Monte Carlo + 27 h-sweep rollouts, each in a fresh
process; estimated 35-40 minutes on the measured machine):

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/evaluate.py \
  --headless --device cuda:0 --num_envs 128 --seed 17 \
  --trajectory_shapes circle --monte_carlo_rounds 5 \
  --h_sweep_rounds 3 --eval_steps 120 \
  --clearance_frame_consistent --isolate_conditions \
  --output_dir PPO2_ROM_B_PhysX/reports/circle_c4_fresh_audit
```

Only after this circle-only audit passes should a final all-shape certification
be run. No PPO training is required for the c4 audit.
