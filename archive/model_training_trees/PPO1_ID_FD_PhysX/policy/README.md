# PPO1_A V3 Policy implementation

The maintained ID/FD policy is the 42-observation/4-action V3 workflow documented in
[the main README](../README.md).

```bash
cd /home/rrlab/rrlab_mulag

# Train with 128 environments and balanced V3 shapes.
./PPO1_ID_FD_PhysX/train_v3.sh

# Visualize a selected checkpoint.
./PPO1_ID_FD_PhysX/visualize_v3.sh /absolute/path/to/model_N.pt
```

The policy has no shape label. It observes reference and measured tool dynamics,
cylinder state and goal error, valve/FD response, XY preview, direction, clearance,
head alignment, and previous actions. Its reward directly penalizes measured tool
acceleration and residual action rate in addition to tracking and safety errors.

The original three-action checkpoint remains an initializer only; it is not the V3
deployment checkpoint.
