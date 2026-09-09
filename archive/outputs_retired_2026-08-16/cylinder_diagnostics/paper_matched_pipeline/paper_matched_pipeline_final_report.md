# Paper-Matched Pipeline: Final Report

**Paper**: [Werner et al. 2025 -- High Precision Hydraulic Excavator Control](https://arxiv.org/pdf/2605.09465)

---

## 1. What Was Copied

| Paper | Our Implementation |
|---|---|
| Two-layer architecture | Upper PPO/task-space + lower cylinder velocity servo |
| Low-level cylinder velocity servo | `_apply_cylinder_velocity_servo()` in `mulag_upper_level_env.py` |
| Feedforward + PID parallel | ID as FF + PID residual |
| Calibration before upper policy | ID/FD fine-tuning + servo tuning before PPO |
| Stage-wise failure isolation | Decision rules A-F in `_stage_failure()` |

## 2. What Was Adapted

| Paper | Our Adaptation |
|---|---|
| MPC path tracking | PPO policy / task-space PD controller |
| Hydraulic LUT (LS/NFC) | Learned neural ID feedforward (HistoryGRU) |
| Cylinder velocity tracking | Cylinder-space servo with ID FF + PID |
| Plant model | ID -> FD -> adapter -> PhysX chain |
| Real-machine calibration | IsaacLab/PhysX calibration + fine-tuning |

## 3. Out of Scope / Forbidden

- PPO directly to valve / PhysX / joint target
- Pulse oracle / BC/DAgger imitation as final controller
- Random gain multiplication without validation
- Silently modifying USD / deleting original checkpoints

---

## 4. Servo Implementation

Control: `e = v_des - v_meas`, `int_e = clamp(int_e + e*dt)`, `de = (e - e_prev)/dt`
`valve_ff = ID(v_des, x, xdot, history)`, `valve_pid = Kp*e + Ki*int_e + Kd*de`
`valve_total = clamp(valve_ff + valve_pid)`, `delta_x_cmd = FD(valve_total, ...)`

FD modes: `prediction_only`, `prediction_plus_feedback_delta`, `bypass_for_diagnostic_only`

---

## 5-6. Results / First Failed Layer

(To be filled after Isaac runtime testing.)

---

## 7. PhysX Audit

See `physx_actuator_tracking_audit.md`

---

## 8. ID/FD Fine-Tuning

See `model_finetune/README.md`

---

## 9. Upper-Level Interface

`v_cyl_desired = v_cyl_base_from_PD_or_DLS + residual_scale * policy_action`

Config: `upper_level_action_mode = "cylinder_velocity_residual"`

---

## 10. Final Metrics

(To be filled)

| Metric | Target |
|---|---|
| EE position RMSE | < 0.05 m |
| Cyl velocity RMSE | < 0.01 m/s |
| Sign accuracy | > 0.95 |

---

## 11. Next Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && cd ~/rrlab_mulag && timeout 600s python kevin_integration/new_observations_pipeline/ppo1/tune_cylinder_velocity_servo.py --num_envs 1 --headless --servo_mode id_ff_plus_cylinder_velocity_pid --fd_usage prediction_only --cylinders 0,1,2,3 --velocities "-0.01,-0.005,0.005,0.01" --kp 6.0 --ki 0.0 --kd 0.0 --valve_cmd_limit 1.0 --cyl_vel_cmd_limit 0.05 --output_dir outputs/cylinder_diagnostics/paper_matched_pipeline
```

---

## 12. Explainability

Upper-level policy/planner commands motion.
Lower-level learned hydraulic servo tracks cylinder velocities.
ID provides feedforward. PID provides feedback.
FD/calibration predicts actuator response. Adapter/PhysX executes.
