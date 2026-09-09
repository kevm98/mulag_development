# Cylinder Velocity Servo Tuning Report

- mode: `id_ff_plus_cylinder_velocity_pid`
- trials: `32`
- pass_count: `0`
- output_csv: `~/rrlab_mulag/outputs/cylinder_diagnostics/standard_control_reset/cylinder_velocity_servo_tuning.csv`
- physx_audit: `~/rrlab_mulag/outputs/cylinder_diagnostics/standard_control_reset/physx_actuator_tracking_audit.md`

## Failed Stage Summary

- A: valve_ff near zero; ID feedforward is weak or out of distribution: 32

## Worst RMSE Trials

| cylinder | v_desired | rmse | sign_accuracy | attenuation | failed_stage |
|---:|---:|---:|---:|---:|---|
| 1 | -0.0500 | 0.106060 | 0.422 | 1.392 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 1 | +0.0500 | 0.103693 | 0.578 | 1.392 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 1 | -0.0200 | 0.094867 | 0.422 | 3.481 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 1 | +0.0200 | 0.093814 | 0.578 | 3.481 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 1 | -0.0100 | 0.093006 | 0.422 | 6.962 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 1 | +0.0100 | 0.092470 | 0.578 | 6.962 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 1 | -0.0050 | 0.092467 | 0.422 | 13.924 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 1 | +0.0050 | 0.092199 | 0.578 | 13.924 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 3 | +0.0500 | 0.074898 | 0.678 | 0.656 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 3 | -0.0500 | 0.074886 | 0.322 | 0.656 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 2 | +0.0500 | 0.069514 | 0.644 | 0.612 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
| 0 | +0.0500 | 0.067358 | 0.411 | 0.415 | A: valve_ff near zero; ID feedforward is weak or out of distribution |
