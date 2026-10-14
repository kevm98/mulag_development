<div align="center">

<sub>MASTER'S RESEARCH · ROBOTICS · 2026</sub>

# Can a hydraulic arm<br>learn to adapt while it works?

**Learning-Based Online Adaptation for Control of a Hydraulic Manipulator**<br>
Kevin Kudakwashe Murera · Robotics Research Lab · RPTU Kaiserslautern-Landau

A simulated MULAG verge mower must keep its cutter on a changing path while terrain and payload disturb the arm. This project asks what learned actuator models and an online payload estimate actually contribute to closed-loop control.

[**Read the thesis**](docs/Kevin_Kudakwashe_Murera_Masterarbeit.pdf) &nbsp;·&nbsp; [**Explore the experiments**](experiments/README.md) &nbsp;·&nbsp; [**Reproduce a run**](docs/REPRODUCE.md)

<br>

<img src="docs/assets/mulag_simulation.gif" alt="Simulation of the MULAG FME 500 arm on a Unimog carrier following a verge" width="900">

<sub>Simulation evidence throughout. This repository makes no claim about deployment on hardware.</sub>

[Front-view rollout](videos/adaptive_rollout_mesh1337_front_view.mp4) · [Back-view rollout](videos/adaptive_rollout_mesh1337_back_view.mp4) · [Video provenance](videos/README.md)

<sub>The front and back videos are separate runs of the same adaptive arm, not two synchronized cameras.</sub>

</div>

---

## The verge mowing challenge

**Tracking is the objective. Terrain is the disturbance.** The truck moves along the road while a hydraulic arm reaches over the embankment. Its cutter must follow a target line and stay near the ground as the terrain changes. The R and IF policies are evaluated over a 26 m path, 11 trajectory families, 2–7 km/h, and three terrain meshes. [Task and pipeline overview](PPO_ROM/README.md).

The research follows two questions:

1. How does the actuator representation affect closed-loop control when the task and scoring conditions are held fixed?
2. When does estimating payload online help a frozen controller, and when does it cost more than it gains?

## Research principles

| Principle | What it means in this project |
|---|---|
| **Pair the conditions** | Compare the same mesh and trajectory family. Report paired differences, sign counts, and paired bootstrap intervals. |
| **Verify the plant** | Confirm that a requested payload or other perturbation reached the simulated plant before interpreting a controller response. |
| **Validate the instrument** | Name the checkpoint and scoring horizon, and measure whether an apparent effect exceeds the instrument's variation. |
| **Keep claims bounded** | Do not compare metrics across collections with different state distributions or normalisers. Separate model prediction, closed-loop performance, and hardware claims. |

These rules, including corrections to earlier interpretations, are recorded in the [experiment index](experiments/README.md).

## Three routes from policy to plant

| Stack | Command path | Role in the study |
|---|---|---|
| [**R · PPO-ROM**](PPO_ROM/README.md) | Policy → command ROM → PhysX | Reduced-order actuator path. |
| [**IF · PPO-ID-FD**](PPO_ID_FD/README.md) | Policy → inverse model → forward model → PhysX | Matched task with a different actuator path. |
| [**AF · PPO-AAM-FD**](PPO_AAM_FD/README.md) | Policy → valve command → frozen forward model → PhysX | Within-stack online adaptation against its own frozen base. |

R and IF choose a cylinder increment; AF chooses a valve command. AF is therefore **not ranked against R and IF on task error**. The individual stack READMEs give the runnable details.

## What the experiments show

### 01 · Adaptation helps when the frozen arm has a heavy-payload deficit

On held-out mesh 3141, a three-seed context-training experiment compares the adaptive AF policy with the same frozen base. The base weights remain fixed; only the context input weights are trained. At 4× and 5× payload, pooled paired medians improve cut-height error, lateral error, and ground contact.

| Payload | Cut-height error improvement | Lateral error improvement | Ground-contact gain |
|---|---:|---:|---:|
| **4×** | **71.52 mm** | **71.58 mm** | **+0.12** |
| **5×** | **82.18 mm** | **107.05 mm** | **+0.09** |

Positive values mean adaptive is better. These are paired median differences across environments pooled over **three context-training seeds on one shared frozen base**, not three independent base policies. The pooled 95% intervals exclude zero, but two of three seeds carry the heavy-load benefit; seed 23 significantly worsens cut height at 5×. [Full result and intervals](reports/RESULT_final_seeds_2026-09-07.md).

### 02 · Adaptation also has a boundary

At **3× payload**, the frozen policy already maintains ground contact. Adaptation makes pooled lateral error **22.75 mm worse** (95% paired interval: 17.27–30.04 mm worse), without a supported pooled cut-height gain. The result suggests gating adaptation to conditions where the frozen policy has a measurable deficit; that gate is a next step, not a demonstrated controller. [Three-seed report](reports/RESULT_final_seeds_2026-09-07.md).

### 03 · Both published actuator routes clear the gate suite

The promoted R and IF policies each pass all eight task gates at 2–7 km/h on meshes 1337, 17, and 909. This is a **pass result for each route**, not a claim that one route outperforms the other. [R protocol](PPO_ROM/README.md) · [IF protocol](PPO_ID_FD/README.md).

### 04 · The hydraulic plant has measurable latency

Command-to-differential-pressure onset was measured at **117 ms**, or seven 60 Hz control steps, across three of four amplitude tiers. That transport delay is identifiable even though the valve's full dynamic model is not identifiable from the same data. [Plant characterisation and method notes](experiments/actuator_representation_and_dynamic_response/README.md).

## Follow the evidence

| Start here | What you will find |
|---|---|
| [Full thesis](docs/Kevin_Kudakwashe_Murera_Masterarbeit.pdf) | The research question, model and controller design, experiments, and limits. |
| [Experiment index](experiments/README.md) | Methods, paired-comparison rules, corrections, and Chapter 7 evidence map. |
| [Final adaptation report](reports/RESULT_final_seeds_2026-09-07.md) | Three seeds, paired intervals, failure cases, and explicit limitations. |
| [Pipeline READMEs](PPO_ROM/README.md) | Commands and checkpoints for each stack; follow the R, IF, and AF links above. |
| [Reproduction guide](docs/REPRODUCE.md) | Setup, a shipped-policy viewer, preflight, and scoring entry points. |

## Citation

```bibtex
@mastersthesis{murera2026online,
  author = {Kevin Kudakwashe Murera},
  title  = {Learning-Based Online Adaptation for Control of a Hydraulic Manipulator},
  school = {Robotics Research Lab, Department of Computer Science, University of Kaiserslautern-Landau},
  year   = {2026},
  type   = {Master Thesis}
}
```
