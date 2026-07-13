# Stop-and-Mow Rebuild References

**Date:** 2026-07-13

## Internal code references (verified)

| Source | Type | Relevant finding | Affected design decision |
|---|---|---|---|
| `ID_FD_pipline/dls_solver/tests/stop_and_mow_gate.py` | Verified gate | Cosine-turnaround sweeps, parking brake, ground-Jacobian pinv seeds, head-down cylinder-4 reservation, separate truck advance phase | Baseline architecture for stop-mow state machine |
| `ID_FD_pipline/dls_solver/ground_paths.py:arc()` | Source code | Graph-diameter path through ground band; `memory.md` says not valid as runtime IK seed | Must NOT be used as PPO trajectory; use verified shapes instead |
| `ID_FD_pipline/models/ee_jacobian_ground_truck3x4.json` | Cached measurement | Jacobian J[:,:3] rank 3, condition 3.85; cylinder 4 can be reserved | Primary Jacobian for arm-only position DLS |
| `ID_FD_pipline/tasks/cylinder_env.py:_compute_ee_pos_local()` | Source code | Truck `base_link` frame, not articulation root; cutter offset -0.79 m | Frame conventions for all trajectory goals |
| `ID_FD_pipline/dls_solver/ee_cascade.py:cylinder_goal_seeded()` | Source code | Map seed + bounded measured-EE DLS correction | How trajectory seeds become cylinder goals |
| `ID_FD_pipline/dls_solver/kinematics.py:ee_xyz_jacobian_stepped()` | Source code | Stepped-PhysX Jacobian; solves closed-loop linkage | Only valid kinematics measurement method |
| `ID_FD_pipline/ppo1/tests/zero_action_contract.py` | Test | Zero action == deterministic baseline | Must be preserved and extended for new shapes |
| `ID_FD_pipline/docs/stop_and_mow_report.md` | Engineering report | Arm-only works; multi-station vehicle transition NOT repeatable | Phase F must fix vehicle before multi-station PPO |
| `memory.md` | Project memory | Max verified spans: 1.00 m line, 0.80 m sweep/semicircle; arc() branches invalid | Shape envelope limits |
| `memory.md` | Project memory | Correct operational architecture: PARK → ARM WORK → RETURN/LIFT → TRUCK ADVANCE → BRAKE → REPEAT | State machine phases |
| `teacher_student/.../commands.py:CustomUniformPoseCommand` | Reference | Converts world target to robot base frame; truck-frame goal anchoring | Frame conversion pattern |
| `teacher_student/.../observations.py:normalized_unimog_pose` | Reference | Normalized truck pose/velocity observations | Observation design |
| `teacher_student/.../rewards.py:tcp_height_to_terrain` | Reference | Left/right cutter height sensing | Ground-following reward |
| `ID_FD_pipline/ppo1/env.py:_apply_truck_speed()` | Source code | Direct wheel velocity write, teacher_student-style | Truck actuation (verify vs parking brake approach) |

## IsaacLab documentation and source references

| Source | Type | Relevant finding | Affected design decision |
|---|---|---|---|
| `isaaclab.envs.DirectRLEnv` | API | `_pre_physics_step`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx` lifecycle | PPO env structure must follow this contract |
| `isaaclab.utils.math.quat_apply_inverse` | API | Transform world→local frame | EE position in truck frame |
| `isaaclab.envs.mdp.actions` | API | Joint position/velocity action interfaces | Must NOT use for scored control (ID/FD pipeline required) |
| `isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg` | API | PPO configuration: `init_noise_std`, `entropy_coef`, `clip_param`, `desired_kl` | PPO hyperparameter choices |
| `isaaclab.markers.VisualizationMarkers` | API | SPHERE_MARKER_CFG for trajectory visualization | Viewer debugging |
| `isaaclab.assets.Articulation.write_joint_velocity_to_sim` | API | Wheel velocity drive | Truck propulsion (vs position brake) |
| `isaaclab.assets.Articulation.write_joint_stiffness_to_sim` | API | Parking brake via high stiffness | Truck braking during mowing |
| `isaaclab.assets.Articulation.set_joint_position_target` | API | Wheel position hold | Alternative parking brake |

## Literature references

| Source | Type | Relevant finding | Affected design decision |
|---|---|---|---|
| Johannink et al., "Residual Reinforcement Learning for Robot Control", ICRA 2019 | Conference paper | Action scale 0.01-0.1 × base output; observation includes base controller internals; reward for tracking + smoothness | PPO action bounds and observation design |
| Silver et al., "Residual Policy Learning", arXiv:1812.06298, 2018 | Preprint | Residual on top of deterministic controller; zero-initialized actor mean | Zero-action contract; actor initialization |
| Nakamura & Hanafusa, "Inverse Kinematic Solutions With Singularity Robustness for Robot Manipulator Control", 1986 | Journal | Damped Least Squares formulation | DLS controller in `cylinder.py` and `ee_cascade.py` |
| Wampler, "Manipulator Inverse Kinematic Solutions Based on Vector Formulations and Damped Least-Squares Methods", 1986 | Journal | DLS damping parameter selection | `damping_smax_ratio` in CylinderDLSConfig |
| Maciejewski & Klein, "Obstacle Avoidance for Kinematically Redundant Manipulators in Dynamically Varying Environments", 1985 | Journal | Task-priority framework | Head-down as secondary task, position as primary |
| Peng et al., "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills", SIGGRAPH 2018 | Conference | Reference-state initialization; phase-indexed policies | Phase-aware observations; state-machine integration |
| Rudin et al., "Advanced Skills through Multiple Adversarial Motion Priors", 2022 | Preprint | Skill chaining; hierarchical RL | Phase-based reward decomposition |
| Craig, "Introduction to Robotics: Mechanics and Control", 3rd ed., 2004 | Textbook | Cosine time-scaling for smooth motion | Turnaround time parameterization |
| Biagiotti & Melchiorri, "Trajectory Planning for Automatic Machines and Robots", 2008 | Textbook | Quintic polynomial time scaling; jerk limits | Optional upgrade from cosine to minimum-jerk |
| Hwang et al., "A Generalized Framework for Mobile Manipulation", 2020 or earlier | Survey | Stop-and-go vs continuous mobile manipulation | Architecture choice: parked mowing is valid for agricultural robots |

## Design decisions supported by references

1. **Residual RL over deterministic controller** (Johannink 2019, Silver 2018): PPO outputs bounded corrections, not full control.
2. **DLS for hydraulic manipulator** (Nakamura 1986): Rectangular Jacobian, damping handles weak directions.
3. **Head-down as secondary task** (Maciejewski 1985): Cylinder-4 null-space projection for orientation constraint.
4. **Cosine turnaround time scaling** (Craig 2004): `h(τ) = 0.5(1 - cos(πτ))` for smooth reversals.
5. **Phase-aware observations** (Peng 2018): One-hot phase + normalized progress enables phase-conditioned policy.
6. **Parked mowing** (Hwang survey): Stop-and-go is architecturally valid for precision agricultural tasks.
7. **Stepped-PhysX Jacobian** (proprietary to this plant): Closed-loop linkage requires physics stepping; analytic Jacobian is invalid.
