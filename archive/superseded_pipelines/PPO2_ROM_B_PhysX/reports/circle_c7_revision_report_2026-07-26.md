# ROM-B circle c7 revision

Date: 2026-07-26

## Reason

The valid 32-row c4 circle audit failed only acceleration:

- Monte Carlo seed 300020: 41.8977 m/s2
- Monte Carlo seed 300035: 41.4589 m/s2
- h=0.5 seed 400020: 40.3194 m/s2
- h=1.2 seed 400030: 40.3251 m/s2

Every termination confidence test and every other physical gate passed.

Fresh diagnostics localized all four acceleration failures to the same geometric
circle region: bins 2-4 (`s=0.691..1.728 m`), with bin 3 reaching 65.15 m/s2.

## Change

c7 preserves every c4 Cartesian point and cylinder command. It changes only the
timing coordinate over bins 2-4:

- bin 2 incremental speed scale: 0.9045;
- bin 3 incremental speed scale: 0.85;
- bin 4 incremental speed scale: 0.9624;
- every other region: unchanged from c4.

The minimum total speed relative to the original c3 timing is 0.6916. The circle
geometry, PPO checkpoint, inner controller, hydraulic ROM, and deterministic
truck are unchanged.

## Seed-17 regression screen

| condition | c4 acceleration | c7 acceleration | c7 clearance | c7 termination |
|---|---:|---:|---:|---:|
| MC round 1 | 41.90 | 39.29 | 0.09794 m | 0/128 |
| MC round 4 | 41.46 | 39.76 | 0.09909 m | 0/128 |
| h=0.5 round 1 | 40.32 | 37.50 | 0.10950 m | 0/128 |
| h=1.2 round 3 | 40.33 | 37.51 | 0.12311 m | 0/128 |
| h=0.4 round 3 | 37.88 | 39.88 | 0.10781 m | 0/128 |
| h=1.0 round 3 | 38.20 | 38.52 | 0.10375 m | 0/128 |

All point gates pass, but the worst acceleration margin is only 0.118 m/s2.
These seeds are now development data and cannot certify c7.

c6 was an intermediate schedule and still failed MC round 4 and h=1.2 by about
0.2 m/s2; it was deleted. c7 is now the ROM-B and ROM-C default.

## Verification

- c7 SHA-256: `c5fcfb249cc7d0be29e6c13340b74953f64ca5ed090472989ab798ea54aa603b`
- 78 tests passed.
- New evaluation reports bind the trajectory path and SHA-256.

## Independent audit

Use a new base seed because seed 17 influenced c7. The predeclared independent
base seed is 29. This produces 32 fresh processes and should take roughly 35-40
minutes on the measured machine.

```bash
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_B_PhysX/scripts/evaluate.py \
  --headless --device cuda:0 --num_envs 128 --seed 29 \
  --trajectory_shapes circle --monte_carlo_rounds 5 \
  --h_sweep_rounds 3 --eval_steps 120 \
  --clearance_frame_consistent --isolate_conditions \
  --trajectory_library_path \
    PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_circle_c7_retime.json \
  --output_dir PPO2_ROM_B_PhysX/reports/circle_c7_seed29_audit
```

No further timing changes should be made from seed-17 results. c7 is accepted or
rejected by the independent seed-29 audit as a complete 32-row set.
