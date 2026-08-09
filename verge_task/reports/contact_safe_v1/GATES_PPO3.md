| cell | 17-gate | strict | contact | post-acq | loss ev | lifts | eject | hard | takeoff | airb | maxgap | maxF | maxlead | align | accel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo3_boom_sweep_h1.0 | PASS | fail | 0.9012 | 0.9047 | 703 | 149 | 10 | 1014 | 415 | 251 | 1.1799 | 100806 | 0.0252 | 0.9744 | 78.7 |
| ppo3_lane_change_h1.0 | PASS | fail | 0.9270 | 0.9303 | 580 | 142 | 11 | 811 | 328 | 36 | 0.3999 | 96328 | 0.0171 | 0.9725 | 65.7 |
| ppo3_mixed_balanced | PASS | fail | 0.9390 | 0.9443 | 499 | 93 | 14 | 858 | 291 | 35 | 0.2714 | 108989 | 0.0196 | 0.9854 | 62.8 |
| ppo3_mixed_h1.0 | PASS | fail | 0.9281 | 0.9315 | 599 | 140 | 9 | 896 | 342 | 36 | 0.2972 | 107456 | 0.0172 | 0.9760 | 66.0 |
| ppo3_mixed_mesh4242 | fail | fail | 0.9326 | 0.9410 | 487 | 142 | 10 | 936 | 306 | 38 | 0.4185 | 119077 | 0.0163 | 0.9386 | 56.9 |
| ppo3_mixed_mesh909 | PASS | fail | 0.9299 | 0.9362 | 512 | 126 | 8 | 850 | 295 | 48 | 0.3906 | 123585 | 0.0153 | 0.9885 | 57.0 |
| ppo3_s_curve_h1.0 | fail | fail | 0.9233 | 0.9268 | 725 | 170 | 11 | 945 | 418 | 31 | 0.3636 | 111806 | 0.0170 | 0.9472 | 79.3 |
| ppo3_s_curve_mesh909 | fail | fail | 0.9236 | 0.9274 | 630 | 136 | 14 | 1002 | 366 | 36 | 0.3594 | 243285 | 0.0162 | 0.9312 | 67.7 |

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
