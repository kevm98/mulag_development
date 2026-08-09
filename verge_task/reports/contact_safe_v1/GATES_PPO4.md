| cell | 17-gate | strict | contact | post-acq | loss ev | lifts | eject | hard | takeoff | airb | maxgap | maxF | maxlead | align | accel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo4_boom_sweep_h1.0 | fail | fail | 0.8689 | 0.8744 | 799 | 202 | 10 | 983 | 483 | 58 | 0.4211 | 123887 | 0.0156 | 0.9415 | 100.5 |
| ppo4_lane_change_h1.0 | fail | fail | 0.8655 | 0.8709 | 558 | 265 | 4 | 663 | 368 | 46 | 0.4820 | 107992 | 0.0225 | 0.9488 | 69.0 |
| ppo4_mixed_balanced | fail | fail | 0.9009 | 0.9131 | 421 | 176 | 2 | 680 | 256 | 38 | 0.4157 | 170498 | 0.0208 | 0.9504 | 63.5 |
| ppo4_mixed_h1.0 | fail | fail | 0.8852 | 0.8908 | 581 | 188 | 4 | 744 | 354 | 44 | 0.4454 | 110310 | 0.0193 | 0.9508 | 75.8 |
| ppo4_mixed_mesh4242 | fail | fail | 0.9099 | 0.9199 | 432 | 169 | 4 | 595 | 250 | 31 | 0.4234 | 121309 | 0.0161 | 0.9466 | 58.5 |
| ppo4_mixed_mesh909 | fail | fail | 0.9026 | 0.9135 | 457 | 178 | 4 | 646 | 281 | 38 | 0.3565 | 87642 | 0.0157 | 0.9589 | 63.3 |
| ppo4_s_curve_h1.0 | fail | fail | 0.8877 | 0.8935 | 606 | 202 | 9 | 809 | 377 | 42 | 0.4454 | 105760 | 0.0193 | 0.9562 | 79.0 |
| ppo4_s_curve_mesh909 | fail | fail | 0.8940 | 0.9049 | 568 | 182 | 13 | 825 | 361 | 41 | 0.3708 | 117445 | 0.0167 | 0.9092 | 76.1 |

**Promotable across every cell: False**

| blocker | cells |
|---|---|
| visible_lift_events | 8 |
| impact_ejection_events | 8 |
| hard_impacts | 8 |
| consecutive_airborne_steps | 8 |
| airborne_gap | 8 |
| post_acquisition_contact | 8 |
| takeoff_events | 8 |
| contact_force_peak | 8 |

teacher_mode across cells: ['off']
direct_target_contact_demo across cells: [False]
