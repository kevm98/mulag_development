# Inverse-dynamics model

The runtime deliberately retains the original cylinder inverse-dynamics
checkpoint:

```text
kevin_integration/new_observations_pipeline/ppo1/Models_cylinder /id_cylinder_space /runs_id_cylinder/id_cylinder_prevcmd_delta_v2/best_id_cylinder.pt
```

Rank-promoting ID candidates were rejected because their validation MSE
regressed. Runtime continuity is provided by evaluating
`ID(target) - ID(zero target)`, which passed the deterministic rank and
functional controller gates.
