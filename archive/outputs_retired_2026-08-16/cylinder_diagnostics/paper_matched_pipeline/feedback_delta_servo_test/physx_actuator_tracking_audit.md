# PhysX Actuator Tracking Audit

Official IsaacLab actuator docs used for this checklist:
- https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html
- https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html

IsaacLab states that implicit actuators are physics-engine actuators: desired position or velocity targets are tracked by the physics engine, and stiffness/damping are written into PhysX. Velocity limits can also constrain achieved velocity if effort is insufficient.

## Runtime Configuration

- low_level_servo_mode: `cylinder_velocity_pid`
- cylinder_dynamics_model: `fd`
- use_sequence_cylinder_models: `True`
- sim.dt: `0.00833333333`
- decimation: `2`
- env step_dt: `0.0166666667`
- sequence_model_dt: `0.05`
- sequence_decimation: `3`
- command write path: `robot.set_joint_position_target(self._cylinder_pos_target, joint_ids=self._cylinder_joint_ids)` every `_apply_action()` call.

## Actuator Groups

- `drehzapfen_cylinder`: `ImplicitActuator`, joint_names_expr=['Drehzapfen_joint']
- `ausleger_i_cylinder`: `ImplicitActuator`, joint_names_expr=['Ausleger_I_joint']
- `ausleger_ii_cylinder`: `ImplicitActuator`, joint_names_expr=['Ausleger_II_joint']
- `messerkopf_schwenk_cylinder`: `ImplicitActuator`, joint_names_expr=['Messerkopf_Schwenk_joint']
- `messerkopf_passive_coupled`: `ImplicitActuator`, joint_names_expr=['Messerkopf_joint']
- `mulag_wheels`: `ImplicitActuator`, joint_names_expr=['Wheel_Front_Left_joint', 'Wheel_Front_Right_joint', 'Wheel_Rear_Left_joint', 'Wheel_Rear_Right_joint', 'Wheel_Front_Left_Steering_joint', 'Wheel_Front_Right_Steering_joint']

## Cylinder Joint Runtime Data

| cylinder | joint_id | pos_limit | stiffness | damping | effort_limit | velocity_limit |
|---|---:|---|---:|---:|---:|---:|
| Drehzapfen_joint | 2 | [-0.08701, 0.26795] | 1.5e+07 | 1e+06 | 1e+08 | 3.40282e+38 |
| Ausleger_I_joint | 3 | [-0.15138, 0.17253] | 3e+07 | 3e+06 | 5e+07 | 3.40282e+38 |
| Ausleger_II_joint | 11 | [-0.16003, 0.14034] | 1e+06 | 500000 | 1e+06 | 3.40282e+38 |
| Messerkopf_Schwenk_joint | 16 | [-0.07426, 0.09555] | 1e+06 | 500000 | 100000 | 3.40282e+38 |

## Audit Findings

- Actuator command mode used by this env: position targets for cylinder joints.
- Target commands are written every env step in `_apply_action()`.
- Servo FD/adapter increments are logged in `cylinder_velocity_servo_tuning.csv` as `fd_delta_abs_mean`, `actual_delta_abs_mean`, and `target_clip_frac`.
- If `valve_total_abs_mean` is high but `fd_delta_abs_mean` is tiny, FD is the bottleneck.
- If FD delta is visible but `actual_delta_abs_mean` is tiny or `target_clip_frac` is high, inspect PhysX drive stiffness/damping, velocity limits, effort limits, and position-target clipping.
