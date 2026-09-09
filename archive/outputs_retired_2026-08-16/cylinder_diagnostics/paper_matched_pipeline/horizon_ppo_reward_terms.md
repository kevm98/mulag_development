# Horizon PPO Reward Terms

Reward adapter:
`kevin_integration/new_observations_pipeline/ppo1/rewards_adapter.py`

The reward is computed once per horizon step after the smooth PhysX ramp.

| Term | Meaning |
|---|---|
| `progress` | `error_before - error_after` in TCP base-frame distance to target |
| `position_tracking` | exponential bonus `exp(-error_after / position_tracking_std)` |
| `close_target` | tanh close-target bonus `1 - tanh(error_after / position_tracking_std)` |
| `yaw_pitch` | currently yaw tolerance penalty against command yaw |
| `tcp_ahead` | TCP forward position relative to arm base, clipped before reward contribution |
| `action_rate` | squared PPO action change across horizons |
| `joint_velocity` | arm joint velocity squared penalty |
| `hydraulic_consistency` | `||delta_x_meas_H - delta_x_cmd_H||` |
| `guard_fraction` | mean per-cylinder workspace/cab/soft-limit guard activation |

Current scalar combination:

```text
2.0 * progress
+ 1.0 * position_tracking
+ 0.5 * close_target
- 0.25 * yaw_pitch
+ 0.02 * clipped_tcp_ahead
- 0.05 * action_rate
- 0.001 * joint_velocity
- horizon_hydraulic_consistency_weight * hydraulic_consistency
- horizon_guard_penalty_weight * guard_fraction
```

Default weights in `MulagUpperLevelEnvCfg`:

- `horizon_hydraulic_consistency_weight = 5.0`
- `horizon_guard_penalty_weight = 0.5`

Explicitly excluded:

- per-step velocity sign reward
- direct valve reward shaping
- camera/ray/height-map obstacle reward terms
