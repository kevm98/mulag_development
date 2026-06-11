# Full Pipeline Preflight Contract

Status: decision-tree preflight implemented in the environment and invoked by training.

## User-Run Entry Points

Preflight only:

```bash
conda activate env_isaaclab
cd ~/rrlab_mulag
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py --num_envs 1 --max_iterations 0 --preflight_only
```

PPO smoke after preflight passes:

```bash
conda activate env_isaaclab
cd ~/rrlab_mulag
./rrlab.sh -p kevin_integration/scripts/train_mulag_adaptive_rl.py --num_envs 1 --max_iterations 20
```

## Decision Tree

Stage 1: no-control stability and reset behavior.

Stage 2: direct actuator transmission.

Stage 3: reference generation.

Stage 4: ID only.

Stage 5: ROM only.

Stage 6: full chain.

## Output Labels

The preflight returns one of:

- `BLOCKED_REFERENCE_TOO_SMALL`
- `BLOCKED_ID_COMMAND_TOO_SMALL`
- `BLOCKED_ID_NORMALIZATION`
- `BLOCKED_ROM_OUTPUT_TOO_SMALL`
- `BLOCKED_MULTIRATE_ATTENUATION`
- `BLOCKED_MAPPING_SCALE`
- `BLOCKED_TARGET_ACCUMULATION`
- `BLOCKED_DRIVE_AUTHORITY`
- `BLOCKED_LINKAGE_TRANSMISSION`
- `READY_FOR_PPO`

## Implemented Gates

- Direct target pulse must produce measured cylinder motion and mechanical family motion.
- Scripted reference motion must exceed the reference threshold.
- Desired qddot must exceed a non-noise threshold.
- ValveCmd mean magnitude must exceed the ID-command threshold.
- ROM 10 ms and 50 ms deltas must exceed motion thresholds.
- Accumulated target motion must exceed target threshold.
- Measured cylinder motion must exceed cylinder threshold.
- Mechanical linkage motion must exceed linkage threshold.
- Clamp fraction must remain below the configured limit.
- Non-finite ID/ROM/control values block PPO.
- The preflight result includes `static_rom_command_coupling`, currently `ValveCmd-independent deployed correction`, so ValveCmd-compatible ID output is not mistaken for proof of ROM command authority.

## Requires Runtime

The decision labels are only meaningful after the user runs the preflight in Isaac. Static code cannot prove visible arm motion or drive authority.

## Current Static Blocker

The active final causal linear ROM is statically ValveCmd-independent. If the preflight starts from rest and measured ExtensionVelocity remains near zero, the expected decision is `BLOCKED_ROM_OUTPUT_TOO_SMALL` or a downstream target/drive label caused by very small ROM deltas. Do not start PPO unless the runtime preflight returns `READY_FOR_PPO`.
