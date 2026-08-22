# Experiments

Each experiment is judged by one question: does it defend a word in the thesis
title, *Learning-Based Online Adaptation for Control of a Hydraulic Manipulator*?

| | Experiment | Covers | Defends | Status |
|---|---|---|---|---|
| [E1](E1/) | Closed-loop task performance | R · IF · AIF | Control | rollouts done |
| [E2](E2/) | Plant characterisation — delay, nonlinearity | R · IF · AIF | Hydraulic Manipulator | delay done |
| [E3](E3/) | Mid-episode plant change | R · IF · AIF | Online Adaptation | running |
| [E4](E4/) | Latent ablation | R · IF · AIF | Online Adaptation | model level done |
| [E5](E5/) | Robustness degradation curve | R · IF · AIF | Hydraulic Manipulator | to run |

R = PPO-ROM · IF = PPO-ID-FD · AIF = PPO-(AAM)-ID-FD

Each folder holds `code/`, `results/`, `figures/` and a README stating the
method, the result and the traps specific to that experiment.

## Rules that apply to every experiment

- **Paired comparisons only.** Report the sign count over the paired unit
  alongside the mean; a mean can be carried by one outlying pair, a unanimous
  sign cannot. Uncertainty by paired bootstrap, reported as a range.
- **Never compare numbers across collections**, even for the same estimand — the
  state distribution and normaliser differ. Retrain the baseline instead.
- **Name the checkpoint and the scoring horizon** in every closed-loop result.
- **Validate the instrument before attributing a result to the system.**

Material predating this structure is in `legacy/`.
