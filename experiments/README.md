# Experiments

Every experiment here answers one question: does it defend a word in the thesis
title, *Learning-Based Online Adaptation for Control of a Hydraulic Manipulator*?

Each folder holds `code/`, `results/`, `figures/` and a README stating the
method, the result, and the traps specific to that experiment.

Stack abbreviations: **R** = PPO-ROM · **IF** = PPO-ID-FD · **AF** = PPO-AAM-FD.

## What each experiment is

| | Question it answers | Stacks | Defends | In the thesis |
|---|---|---|---|---|
| [E1](E1_tracking_and_control_effort/) | How well does each stack track the verge under nominal conditions? | R · IF | Control | §7.2, §7.3.5 |
| [E2](E2_transport_delay_and_identifiability/) | How much dead time does the actuation chain carry, and is the plant identifiable from the data it produces? | plant property | Hydraulic Manipulator | §7.3.1 |
| [E3](E3_mid_episode_plant_change/) | Does a history-conditioned residual improve control when the load changes mid-episode, and is the improvement adaptation? | AF, within-stack | Online Adaptation | §7.5 |
| [E4](E4_latent_ablation/) | Does what occupies the latent slot change the closed loop? | AF, one checkpoint | Online Adaptation | Appendix E only |
| [E5](E5_plant_robustness_envelope/) | How far can the plant be degraded before a stack fails its acceptance gates? | R (ladder) · R · IF (audit) | Hydraulic Manipulator | §7.4, §7.4.1 |
| [E6](E6_latent_bandwidth_sweep/) | Is the latent's harm its time-variation or its content? | AF, one checkpoint | Online Adaptation | not cited |

## Status

- **E1** complete for the matched pair. The AF rows were scored on the earlier
  ID-adapter architecture and are withdrawn, not reported (§7.1.3).
- **E2** dead time complete. The nonlinearity probe was never run.
- **E3** complete, including the correction. The first dose sweep confounded the
  mid-episode change with static mass, because `MULAG_MASS_BIMODAL=1.0,X` puts
  half the fleet at X from spawn. The uniform-step rerun isolates it and
  resolves an adaptation component on both metrics. See `E3/results/`.
- **E4** model level complete. Its arms ran on the retired architecture, so they
  survive as a recorded measurement in Appendix E rather than as a result about
  the deployed stack.
- **E5** the factor sweep was withdrawn for want of a positive control: two of
  three stacks are inert to the hydraulic factors, and what does act is clamped.
  The actuator-authority ladder replaces it and is what §7.4 reports.
- **E6** complete, verdict negative, and nothing in the thesis cites it. Kept as
  the record of a question that was asked and answered.

## Rules that apply to every experiment

- **Paired comparisons only.** Report the sign count over the paired unit
  alongside the mean; a mean can be carried by one outlying pair, a unanimous
  sign cannot. Uncertainty by paired bootstrap, reported as a range.
- **A marginal median is not an effect size.** Between two runs identical by
  construction the marginal median moved 6.386 to 7.361 mm. Only the paired
  difference survives contact non-determinism.
- **Never compare numbers across collections**, even for the same estimand. The
  state distribution and the normaliser differ. Retrain the baseline instead.
- **Name the checkpoint and the scoring horizon** in every closed-loop result.
- **Verify the perturbation reached the plant** before analysing. Read
  `mass_scale_mean` from the trace or `factors.mass_scale` from the report. E5
  withdrew a conclusion and E3 corrected one for want of this check.
- **Validate the instrument before attributing a result to the system.** The
  measured false-positive rate is one in three on cut height, which sets a
  ±10 % floor below which a single pair of runs cannot resolve an effect.

## Other folders

| | What it is |
|---|---|
| `AAM_S/` | Offline AAM sweeps over history length and smoothing. Feeds §5.5. |
| `RMA_PPO/` | The CPU surrogate environment and robustness harness used before the IsaacLab pipeline existed. |
| `control_latent/` | Early latent-conditioning probes, superseded by E4 and E6. |
| `legacy/` | Material predating this structure. Nothing here is cited. |
