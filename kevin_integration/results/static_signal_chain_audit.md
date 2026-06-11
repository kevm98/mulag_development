# Static Signal Chain Audit

Status: static audit completed outside Isaac. Runtime motion remains to be validated by the user in `env_isaaclab`.

## Active Runtime Chain

The active path is:

`reference/PPO action -> qddot_cmd[4] -> AAM history -> z_arm_hat[24] -> ID -> ValveCmd[4] -> final causal ROM -> DeltaExtension_simscape[4] per 0.05 s -> nominal sign adapter -> DeltaJoint_isaac[4] per 0.05 s -> five 0.01 s increments -> accumulated prismatic targets -> Isaac position drives -> linkage constraints`.

FD effort actuation is disabled for the hydraulic ROM task. SD feedback into ROM remains disabled.

## Proven Statically

- Active hydraulic channel order is `Drehzapfen_joint`, `Ausleger_I_joint`, `Ausleger_II_joint`, `Messerkopf_Schwenk_joint`.
- Mechanical state also includes `Messerkopf_joint`; the ROM still outputs exactly four cylinder deltas.
- ID dynamic input contract is 19 features: `q[5], qdot[5], qddot_cmd[5], dP[4]`.
- ID checkpoint declares `dynamic_dim=19` and `window_size=100`.
- ID qddot scaler slots are `feature_std[10:14] = [4.087958, 2.653908, 15.052602, 19.057526]`.
- ID dP slots are `feature_mean[15:19] = [-33525270, 13099496, 433658, 5325178]` Pa and `feature_std[15:19] = [67825632, 64911044, 50037828, 69429288]` Pa.
- Final ROM export metadata is `selected_model_type=causal_linear_correction`, causal, `dt=0.05`, 26 input features, and four DeltaExtension outputs.
- The deployed final causal linear correction uses only `dP`, `Extension`, `ExtensionVelocity`, `JointPos`, and `JointVel` per cylinder. `ValveCmd` is transported in the 26-feature runtime vector but is not a coefficient feature in the deployed correction.
- Causal ROM training target median absolute 50 ms DeltaExtension is approximately `[0.00616, 0.00777, 0.02549, 0.01431]` m.
- Training ValveCmd support is broad and signed. Median absolute ValveCmd is approximately `[0.864, 0.907, 0.288, 0.473]`.
- With zero extension velocity and mean state/dP, the deployed ROM correction from rest is approximately `[-0.000192, -0.000014, -0.000132, -0.000134]` m per 50 ms.
- Changing ValveCmd from `[-1,-1,-1,-1]` to `[1,1,1,1]` at the same state produced exactly zero output difference in pure-Python inference.
- Coordinate delta mapping in active nominal-sign mode is sign-only for deltas and velocities. Current signs are `[-1, -1, -1, -1]`; magnitude scale is 1.
- Multirate wrapper divides each 50 ms ROM delta by exactly five and applies the cached increment across five 10 ms steps.
- Target accumulation is persistent: `target_next = clamp(target_current + delta_10ms)`.

## Corrected

- Scripted demo episode length is 300 s for the hydraulic ROM task.
- Scripted demo now has a 1 s settle window and 2 s smooth learned-control ramp.
- AAM, ID, FD, SD, and final ROM histories are warm-started from the current settled state instead of long zero-fill transients.
- Scripted reference amplitude and gains were raised upstream so all four hydraulic families receive meaningful qddot demands.
- Desired qddot now includes explicit feedback and sinusoidal feedforward terms, with each term logged.
- Runtime telemetry now logs ID raw/normalized input ranges, z_arm norm, ValveCmd stats, ROM raw/normalized correction features, kinematic/correction/final ROM deltas, mapping signs, accumulated targets, measured cylinder motion, and tracking ratios.
- Reset telemetry now includes `reset_count`, `reset_reason`, `episode_step`, `timeout`, `unsafe_termination`, and `target_reached_termination`.
- Training supports `--preflight_only`, and `--max_iterations 0` exits after preflight before PPO runner creation.

## Requires Isaac Runtime

- Actual reference percent of usable Isaac limits after USD articulation load.
- Whether ValveCmd remains training-compatible under live state, pressure, and z_arm histories.
- Whether ROM output is training-compatible under live ValveCmd and mapped cylinder state.
- Whether prismatic drives track accumulated targets.
- Whether linkage constraints transmit cylinder motion to the arm.
- Whether root drop/jump is only reset/gravity behavior or indicates drive/contact instability.

## Unresolved Until Runtime Logs

- The USD-authored drive stiffness, damping, force limits, velocity limits, and actual target buffer readback cannot be proven without Isaac/USD runtime introspection here.
- The nominal cylinder coordinate mapping is sign verified but not second-pose validated.
- Visible significant motion must not be claimed until the user runs the scripted demo and returns telemetry.

## Static Root Cause

The most important static root cause is now proven: the selected final causal ROM is a transition predictor whose deployed correction is ValveCmd-independent. In the active runtime, dP is fixed to the ROM training mean and SD feedback is disabled. Therefore ID can output large, signed ValveCmd values, but those commands do not directly change the final ROM DeltaExtension. Starting from rest, the kinematic term is also near zero because measured ExtensionVelocity is near zero. The remaining learned correction is sub-millimetre per 50 ms at the mean state.

This explains almost no visible arm movement without changing learned model architecture. The correct runtime response is to expose this bottleneck, gate PPO with preflight, and avoid hiding it with arbitrary DeltaExtension scaling.
