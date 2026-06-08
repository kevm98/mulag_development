# Force-Actuated Cylinder Debug

Status: diagnostic tooling prepared; permanent force-actuated Kevin config is intentionally not created yet.

## Why Direct Effort Failed With Authored Drives

Runtime/USD inspection showed that Kevin's four command joints are actuator-side prismatic cylinder joints, not the final revolute arm joints. `Drehzapfen_joint` is a `PrismaticJoint` with a strong authored linear drive:

- stiffness: `15,000,000`
- damping: `1,000,000`
- maxForce: `100,000,000`
- targetPosition: `0`

The direct effort and actuator-to-arm transmission diagnostics showed motion similar to zero-effort drift for `Drehzapfen_joint`. That is consistent with effort control fighting an authored position drive that holds the prismatic actuator at its target.

## Drive Values Found

Known `Drehzapfen_joint` values are listed above. The new `inspect_cylinder_actuator_drives.py` script must be run inside IsaacLab to collect the same USD and runtime values for:

- `Drehzapfen_joint`
- `Ausleger_I_joint`
- `Ausleger_II_joint`
- `Messerkopf_Schwenk_joint`

Command:

```bash
./rrlab.sh -p kevin_integration/scripts/inspect_cylinder_actuator_drives.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 1 \
  --headless
```

## Force Transmission Before Drive Override

Latest known result for `Drehzapfen_joint`: failed/inconclusive. Positive and negative effort did not produce actuator stroke or expected `Drehzapfen_constraint_joint_*` motion clearly separable from zero drift. Largest mechanical motion appeared in an unexpected family, and body motion was also present during zero effort.

## Diagnostic Drive Override

The transmission diagnostic now supports:

```bash
--diagnostic_disable_actuator_drives \
--diagnostic_drive_stiffness 0.0 \
--diagnostic_drive_damping 100.0
```

This is runtime-only and diagnostic-only. It does not edit the USD, does not change training config, and restores original gains before shutdown when possible.

Recommended first force-transmission command:

```bash
./rrlab.sh -p kevin_integration/scripts/test_actuator_to_arm_transmission.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 1 \
  --steps_per_test 100 \
  --settle_steps 500 \
  --actuator_joint_names Drehzapfen_joint,Ausleger_I_joint,Ausleger_II_joint,Messerkopf_Schwenk_joint \
  --effort_values 100,500,1000,5000 \
  --diagnostic_disable_actuator_drives \
  --headless
```

## Position-Target Transmission Test

The force/effort transmission tests failed even after the runtime drive override, so the next diagnostic is to command
the four prismatic cylinder joints with position targets through the authored drive interface. This is still
diagnostic-only: it does not change reward, learned models, torque scaling, training config, USD, or permanent actuator
configuration.

The position-target test validates the USD linkage if actuator stroke targets produce actuator motion beyond zero drift,
positive and negative offsets move in opposite directions, and the expected mechanical family moves more than unrelated
families while `done_frac` remains `0`. In that case, the linkage can transmit cylinder stroke through the authored
interface, and the remaining problem is force actuation or actuator setup.

If position targets also fail to move the expected arm/revolute families beyond zero drift, the linkage/constraint setup
itself needs asset repair before learned control, PPO training, or a force-actuated config can be justified.

Commands:

```bash
./rrlab.sh -p kevin_integration/scripts/test_actuator_position_target_transmission.py \
  --task Kevin-Mulag-AdaptiveRL-JointReach-v0 \
  --num_envs 1 \
  --steps_per_test 200 \
  --settle_steps 500 \
  --actuator_joint_names Drehzapfen_joint,Ausleger_I_joint,Ausleger_II_joint,Messerkopf_Schwenk_joint \
  --position_offsets 0.005,0.01,0.02 \
  --headless
```

```bash
python kevin_integration/scripts/analyze_actuator_position_target_transmission_csv.py \
  --csv logs/actuator_position_target_transmission/<timestamp>/actuator_position_target_transmission.csv \
  --reference_body Frontrahmen
```

## Decision: First ROM Integration Uses Cylinder Position Targets

The first reduced-order hydraulic actuator integration should not output raw cylinder force to Isaac. Force transmission
has not passed diagnostics, while position-target transmission has validated the USD linkage through the authored
prismatic cylinder drive interface.

The implemented safe architecture is therefore:

- valve command / reduced hydraulic state
- delay, dead-zone, flow, and pressure-state ROM
- bounded cylinder velocity or stroke increment
- prismatic cylinder position target via `set_joint_position_target`
- USD linkage moves the arm

This is aligned with hydraulic actuator literature at the structural level: delay/dead-zone/nonlinear input maps are
kept explicit, pressure and cylinder velocity are treated as reduced states, and the learned/physics-inspired blocks stay
modular rather than becoming a single black-box correction. The current parameters are conservative placeholders, not
identified physical parameters.

Remaining future work for force actuation:

- repair or reconfigure USD/PhysX transmission so direct cylinder force passes diagnostics
- identify ROM parameters from Simscape/full-order data
- only then test a force-output ROM and `set_joint_effort_target`

## ROM Position-Target Integration Validation

The initial ROM position-target integration smoke test ran successfully: it used the prismatic position-target path,
remained stable, produced no NaNs, and kept `done_frac=0`. That only proves that the software plumbing and USD position
target pathway can run.

The earlier PASS logic was too weak because it judged each positive or negative valve test independently. Some positive
and negative valve commands produced same-direction actuator or mechanical motion, and zero valve also produced nonzero
motion. The stricter validation now compares zero, positive, and negative valve runs for each actuator and valve
magnitude.

Required validation:

- ROM `dx` is nonzero relative to zero drift
- positive and negative valve commands produce opposite-signed ROM `dx`
- actuator target and actual actuator motion separate from zero drift
- actual actuator position tracks the ROM target within the configured tolerance
- expected mechanical family moves beyond zero drift
- NaN guards, done flags, and clamp fractions do not dominate behavior

The current placeholder ROM parameters likely need identification from Simscape/full-order data. Use the synthetic
`--rom_override_dx_per_step` mode only to validate target plumbing; it is not hydraulic physics and must not be used for
training. The next milestone is fitting the reduced pressure/velocity ROM parameters from data and then rerunning the
strict zero/positive/negative validator.

## Feasibility Decision Gate

Force actuation is feasible only if all four cylinder paths pass the direct transmission test:

- positive force moves actuator stroke one way
- negative force moves actuator stroke the other way
- expected mechanical family moves beyond zero drift
- expected mechanical family dominates unrelated families
- body motion is not only global settling
- no NaNs
- `done_frac` remains `0`
- limits remain respected

If the diagnostic drive override still fails, do not create a force-actuated config. The next repair target is USD linkage/constraint setup.

## Config Variant

No Kevin force-actuated robot config variant has been created yet because the diagnostic override has not been proven in IsaacLab.

When diagnostics pass, the safe variant should be Kevin-specific and non-destructive:

- use the same USD
- explicitly target only:
  - `Drehzapfen_joint`
  - `Ausleger_I_joint`
  - `Ausleger_II_joint`
  - `Messerkopf_Schwenk_joint`
- set cylinder stiffness to `0.0`
- set initial damping to `100.0`
- set safe effort/velocity limits based on inspected USD/runtime limits
- preserve non-controlled joints, wheels, steering, and `Messerkopf_joint` behavior as much as possible

Do not use `joint_names_expr=[".*"]` for the force-actuated group.

## Remaining Risks

- Runtime drive override may not fully remove authored USD drive behavior if Isaac/PhysX retains target constraints differently.
- Linkage constraints may not transmit cylinder force into the expected revolute families.
- World body translation can reflect global settling; use zero-drift-relative and reference-body analysis.
- Learned FD outputs are not yet calibrated for direct prismatic cylinder force in Isaac.

## Recommendation

Current recommendation: do not train and do not reconnect learned FD effort yet.

Run:

1. `inspect_cylinder_actuator_drives.py`
2. `test_actuator_to_arm_transmission.py --diagnostic_disable_actuator_drives`
3. `analyze_actuator_to_arm_transmission_csv.py --reference_body Frontrahmen`

Proceed toward a Kevin-specific force-actuated config only if all four actuator paths pass.
