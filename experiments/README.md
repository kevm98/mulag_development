# Experiments

Every experiment answers one question: does it defend a word in the thesis title,
*Learning-Based Online Adaptation for Control of a Hydraulic Manipulator*?

Each folder holds `code/`, `results/`, `figures/` and a README stating the
method, the result, and the traps specific to that experiment.

Stack abbreviations: **R** = PPO-ROM · **IF** = PPO-ID-FD · **AF** = PPO-AAM-FD.

## What is here

Only the experiments the current thesis actually draws on.

| | Question it answers | Stacks | Produces |
|---|---|---|---|
| [E1](E1_tracking_and_control_effort/) | How well does each stack track the verge under nominal conditions, and at what control effort? | R · IF | §7.2 `ch7_e1_paths`, `ch7_e1_tracking`; §7.3.5 `ch7_control_effort` |
| [E2](E2_transport_delay_and_identifiability/) | How much dead time does the actuation chain carry, and is the plant identifiable from the data it produces? | plant property | §7.3.1 delay and step-response tables |

`code/` in each folder regenerates its figures. E1 reads `outputs/e1_traces`,
`outputs/e1_tracking` and `outputs/stage31_openloop`; E2 reads `outputs/e2_chain`
and `outputs/e2_refstep` plus its own `results/*.json`.

## Not here

The §7.4 authority sweep and the §7.5 adaptation result are **not** produced from
this folder. Their figures — `ch7_kappa_stacks` and `ch7_adaptation_forest` — are
built by `PPO_AAM_FD/experiment_scripts/make_kappa_figure.py` and
`make_adaptation_figure.py` from `outputs/kappa_loaded` and `outputs/final`, as
the thesis appendix records.

E3 to E6 were moved to `archive/experiments_stale_2026-09-09/` on 2026-09-09:
none of them feeds a figure or table in the current thesis. E3's mid-episode
plant change was superseded by the `outputs/final` adaptation result, E5's factor
sweep was withdrawn and its authority ladder replaced by the κ sweep, and E4 and
E6 were never cited. They are kept as the record of questions asked and answered.

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
  `mass_scale_mean` from the trace or `factors.mass_scale` from the report.
- **Validate the instrument before attributing a result to the system.** The
  measured false-positive rate is one in three on cut height, which sets a
  ±10 % floor below which a single pair of runs cannot resolve an effect.

## Other folders

| | What it is |
|---|---|
| `AAM_S/` | Offline AAM sweeps over history length and smoothing. Feeds §5.5. |
| `RMA_PPO/` | The CPU surrogate environment and robustness harness used before the IsaacLab pipeline existed. |
| `control_latent/` | Early latent-conditioning probes. Not cited. |
| `legacy/` | Material predating this structure. Nothing here is cited. |
