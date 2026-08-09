# PPO2_A V3 Policy environment

This directory provides the shared hydraulic environment, observations, rewards,
trajectory supervisor, and park-work-relocate runtime used by the canonical
42-observation/4-action V3 policy in `policy_4cyl/`.

The maintained user commands are:

```bash
cd /home/rrlab/rrlab_mulag

# Train PPO2-ROM-PhysX V3.
./mulag_plant/train_large_v3.sh

# Run park, track, stow, straight drive, and next-station visualization.
./mulag_plant/visualize.sh
```

The older three-action actor is an internal transfer/rollback dependency only and is
not a documented training or deployment workflow.

See [the canonical V3 README](../README.md).
