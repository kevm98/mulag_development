# Flat-road / adaptive verge-contact physics report

Date: 2026-07-29

## Implemented task contract

- The truck travels only on a straight, flat, yellow road at 4 km/h.
- The road and green roadside verge are separate physical meshes.
- The truck pose is constrained to the road line and its complete chassis
  attitude is held, while four tire/road contact sensors verify support.
- The front steering joints are commanded to their authored zero-angle state
  rather than a displaced post-settle snapshot. A live axle-orientation check
  now rejects more than 1 degree of front-wheel steering error.
- The authored tire radius and chassis target put all tire bottoms at road
  height; truck grounding is no longer inferred from the misleading root frame.
- The roadside terrain is entirely green and deliberately smooth but
  non-uniform. Three broad hill sections peak nominally at 18, 48 and 27 cm,
  over a gentle 12 cm background rise. Their locations, widths, and heights vary
  by seed; long-wave surface variation grows from 1.5 to 6 cm and maximum
  cross-slope remains 0.005 rad.
- A red visual ribbon is projected onto the exact analytic terrain centreline.
- The cutter uses a real Isaac Lab `ContactSensor`, filtered to the green verge
  collider. Contact is no longer inferred from cutter height.
- The non-PPO visualization controller combines terrain-height feed-forward,
  cutter contact-force feedback, bounded reseating, and lateral line tracking.
- PhysX depenetration is limited to 0.20 m/s so contact settles instead of
  producing repeated high-speed ejection from the rigid terrain.
- The policy observation has 51 channels and includes normalized physical
  cutter contact force. Reward and convergence semantics now reward ground
  contact rather than treating it as a strike.
- The wheel convergence gate uses per-wheel sensor duty cycle, not the minimum
  instantaneous contact fraction from one solver sample.

## Verification

Final four-environment, 600-step headless probe:

- Result: **PASS**
- Speed: 3.99999 km/h for a 4.0 km/h command
- Truck lateral drift: 0.0000038 m maximum
- Chassis tilt: 0.0 degrees maximum
- Tire-bottom error from road z=0: 0.00031 m maximum
- Minimum per-wheel road-contact duty cycle: 0.245
- Cutter contact after acquisition, per terrain: 0.9175, 0.9200, 0.8450, 0.8850
- Maximum cutter centre/path error, per terrain: 0.1392, 0.1531, 0.1403, 0.1374 m
- Terrain height variation encountered: 0.0542 m
- Cutter-origin height adaptation: 0.1254 m
- Height-scan validity: 1.0
- Policy observations and rewards: finite

Machine-readable evidence:
`PPO3_SOLE_PhysX/reports/contact_road_task_probe_4env_final.json`

After increasing the visible but smooth terrain rise, a second four-environment
600-step probe also passed:

- Encountered terrain-height range: 0.1709 m
- Cutter-origin adaptive height range: 0.3346 m across randomized profiles
- Cutter contact after acquisition, per terrain: 1.0000, 0.9975, 0.9550, 0.9300
- Maximum line error, per terrain: 0.1180, 0.0961, 0.0999, 0.1382 m
- Chassis tilt: 0.0 degrees

Updated evidence:
`PPO3_SOLE_PhysX/reports/rising_soft_terrain_4env.json`

The later realistic-height revision replaces the uniform rise with three broad,
seed-varied hills. The tallest analytic section reaches about 0.58 m while its
maximum grade remains about 9.3%. The red path curves smoothly with the boom's
feasible sweep and is projected onto that surface. A 720-step one-environment
run passed with 0.277 m encountered height range, 0.262 m cutter vertical
adaptation, 97.5% contact, 0.149 m maximum / 0.084 m p95 path error, no reset,
4 km/h motion, and zero chassis tilt.

Evidence:
`PPO3_SOLE_PhysX/reports/high_irregular_soft_terrain_1env.json`

## Seven-family trajectory curriculum

The red-path generator now cycles seven deterministic, seed-varied families
across parallel environments: straight, lateral sine, S-curve, smooth random
spline, gradual offset-and-return, height-coupled boom sweep, and lateral chirp.
All paths use the same analytic samples for rendering, controller error, reward,
and policy input. The 54-D observation adds path offsets 0.5, 1.0 and 2.0 m
ahead, allowing PPO to anticipate curvature rather than react only after error.

The six plan-view families use varied moderate terrain for a feasible early
curriculum. The boom-sweep family retains the full 58-cm irregular hill. A
simultaneous seven-environment, 720-step run passed:

| Family | Contact | Maximum centre error |
|---|---:|---:|
| Straight | 99.6% | 0.077 m |
| Lateral sine | 100.0% | 0.222 m |
| S-curve | 99.0% | 0.241 m |
| Random spline | 92.7% | 0.191 m |
| Offset and return | 92.9% | 0.083 m |
| Boom sweep / high terrain | 99.6% | 0.270 m |
| Lateral chirp | 91.7% | 0.165 m |

The 0.30-m centre-error acceptance remains inside the physical mower-head
width. Terrain profiles differ, scan validity is 100%, tire support is sensed,
front steering is straight, and chassis tilt is 0 degrees.

Evidence:
`PPO3_SOLE_PhysX/reports/all_path_families_7env_final.json`

Static verification: Python compilation passed and all 76 PPO3 tests passed.

## Training status

No PPO training was launched after these changes. The pre-fix checkpoint at
`logs/rsl_rl/ppo3_sole_terrain/heldout_convergence_straight_progress_v1/model_paused_390.pt`
is retained only as actor initialization. The long-run command resets its stale
optimizer and critic state, migrates the observation input from 50 to 51
channels, trains 512 environments in bounded blocks, evaluates 128 held-out
environments, checkpoints each block, and stops only when every shared gate
passes.
