# ROM and ID/FD on the corrected stage-6 driving task

Generated 2026-08-11T10:15:02.

Stage 6 (flat ground with raised chunks), path amplitude band
(0.30, 0.45, 0.60, 0.75), truck driving at 4 km/h, 6 held-out meshes,
32 envs, 480 steps, worst mesh reported.

These are the first numbers measured on the task the policies are
actually trained on. Previously PPO trained with a PARKED truck (so the
verge path never advanced), acceptance scored stage 5, and the stage-6
curriculum overwrote the task's feasible amplitude band with a ladder
whose every rung demanded more lateral cutter speed than the cylinders
can deliver. Nothing measured before 2026-08-11 is comparable.

| arm                          | pass |  cut_height |     lateral | ground_cont | head_alignm | hard_impact | tool_accele | cylinder_he | safety_term |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| teacher (DLS, ROM plant)     | no   |      0.0392 |      0.0958 |      0.4198 |      0.9841 |      0.2548 |     62.6237 |      0.0000 |      0.0673 |
| teacher (DLS, ID/FD plant)   | no   |      0.0452 |      0.0537 |      0.2854 |      0.9859 |      0.3074 |     71.6212 |      0.0373 |      0.0779 |
| null (zero action)           | no   |      0.0348 |      0.1583 |      0.5326 |      0.9590 |      0.0235 |     40.9329 |      0.0000 |      0.0585 |
| ROM   distilled              | no   |      0.0172 |      0.0909 |      0.6208 |      0.9862 |      0.1926 |     56.9414 |      0.0000 |      0.0253 |
| ID/FD distilled              | no   |      0.0127 |      0.0654 |      0.7778 |      0.9934 |      0.2301 |     61.9199 |      0.0390 |      0.0148 |
| ROM   PPO (mispriced reward) | no   |      0.0053 |      0.3566 |      0.5847 |      0.9630 |      0.5243 |     53.9297 |      0.0000 |      0.0255 |
| ROM   distilled + PPO        | no   |      0.0055 |      0.0803 |      0.6486 |      0.9918 |      0.1634 |     58.4045 |      0.0004 |      0.0070 |
| ID/FD distilled + PPO        | no   |      0.0044 |      0.0463 |      0.9306 |      0.9872 |      0.2964 |     48.7829 |      0.0392 |      0.0059 |

Thresholds: cut_height <=0.04, lateral <=0.3, ground_contact >=0.7, head_alignment >=0.9, hard_impacts <=0.1, tool_acceleration <=120, cylinder_headroom >=0.005, safety_terminations <=0.08

- **teacher (DLS, ROM plant)** fails: cylinder_headroom, ground_contact, hard_impacts
- **teacher (DLS, ID/FD plant)** fails: cut_height, ground_contact, hard_impacts
- **null (zero action)** fails: cylinder_headroom, ground_contact
- **ROM   distilled** fails: cylinder_headroom, ground_contact, hard_impacts
- **ID/FD distilled** fails: hard_impacts
- **ROM   PPO (mispriced reward)** fails: cylinder_headroom, ground_contact, hard_impacts, lateral
- **ROM   distilled + PPO** fails: cylinder_headroom, ground_contact, hard_impacts
- **ID/FD distilled + PPO** fails: hard_impacts

## Worst-mesh lateral error by path family

The lateral gate reads the worst family. A family that is bad for the
privileged teacher as well is a reachability question, not a training one.

| family             | teacher (DLS, | teacher (DLS, | null (zero ac | ROM   distill | ID/FD distill | ROM   PPO (mi | ROM   distill | ID/FD distill |
|----------------------------------------------------------------------------------------------------------------------------------------------------|
| boom_sweep         |        0.0600 |        0.0434 |        0.1452 |        0.0658 |        0.0371 |        0.2613 |        0.0481 |        0.0351 |
| double_s_curve     |        0.0958 |        0.0537 |        0.1135 |        0.0645 |        0.0520 |        0.2668 |        0.0697 |        0.0449 |
| fourier_meander    |        0.0478 |        0.0448 |        0.0835 |        0.0404 |        0.0482 |        0.3566 |        0.0401 |        0.0386 |
| gaussian_slalom    |        0.0663 |        0.0416 |        0.1317 |        0.0559 |        0.0394 |        0.2767 |        0.0409 |        0.0389 |
| lane_change_hold   |        0.0742 |        0.0522 |        0.1583 |        0.0666 |        0.0654 |        0.2358 |        0.0408 |        0.0392 |
| lateral_chirp      |        0.0573 |        0.0411 |        0.0825 |        0.0403 |        0.0372 |        0.3012 |        0.0475 |        0.0288 |
| lateral_sine       |        0.0664 |        0.0407 |        0.1319 |        0.0415 |        0.0382 |        0.3434 |        0.0385 |        0.0334 |
| offset_return      |        0.0816 |        0.0469 |        0.1537 |        0.0909 |        0.0441 |        0.2106 |        0.0803 |        0.0463 |
| random_spline      |        0.0557 |        0.0498 |        0.1219 |        0.0719 |        0.0509 |        0.3005 |        0.0471 |        0.0375 |
| s_curve            |        0.0576 |        0.0409 |        0.0797 |        0.0427 |        0.0362 |        0.2603 |        0.0404 |        0.0284 |
| straight           |        0.0834 |        0.0442 |        0.1249 |        0.0496 |        0.0367 |        0.1732 |        0.0463 |        0.0351 |

