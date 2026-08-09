# PPO2_A V3 four-cylinder Policy implementation

This is the canonical policy implementation used by PPO2-ROM-PhysX V3.

```text
observations: 42
actions:      4
cylinders 1–3 residual authority: ±2.00 mm
cylinder 4 residual authority:    ±0.25 mm
selected checkpoint: model_607.pt
```

Cylinder 4 controls mower-head pitch. Its smaller authority, stronger filtering, and
separate magnitude/rate penalties prevent the policy from reducing XYZ error by
tilting the tool. All four outputs train jointly; the hydraulic ROM remains the sole
learned actuator model between the policy command and PhysX.

## Visualize the canonical policy

```bash
cd /home/rrlab/rrlab_mulag
./mulag_plant/visualize.sh
```

## Train the canonical V3 policy

```bash
cd /home/rrlab/rrlab_mulag
./mulag_plant/train_large_v3.sh
```

## Verify the four-cylinder contract

```bash
cd /home/rrlab/rrlab_mulag
/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  mulag_plant/policy_4cyl/tests/test_four_cylinder_contract.py
```

The paired promotion probe compares matching environment indices for tracking mean,
tracking p95, p95 tool acceleration, completion, head alignment, and retained truck
drift. The deployment supervisor independently enforces park → track → stow → straight
drive → park and never exposes wheel control to PPO.

See [the main V3 README](../README.md) for canonical artifacts, trajectory dimensions,
and operational details.
