# Experiments

Every experiment answers one question: does it defend a word in the thesis title,
*Learning-Based Online Adaptation for Control of a Hydraulic Manipulator*?

Each folder holds `code/`, `results/`, `figures/` and a README stating the
method, the result, and the traps specific to that experiment.

Stack abbreviations: **R** = PPO-ROM · **IF** = PPO-ID-FD · **AF** = PPO-AAM-FD.

## What is here

One folder per section of thesis Chapter 7, each holding the code that produces
that section's results and the figures it produces.

| Folder | Thesis | Produces |
|---|---|---|
| [closed_loop_task_performance](closed_loop_task_performance/) | §7.2 | `ch7_e1_paths`, `ch7_e1_tracking` |
| [actuator_representation_and_dynamic_response](actuator_representation_and_dynamic_response/) | §7.3 | delay and step-response tables; `ch7_control_effort` |
| [generalisation_robustness_and_authority_margin](generalisation_robustness_and_authority_margin/) | §7.4 | `ch7_kappa_stacks` |
| [online_adaptation](online_adaptation/) | §7.5 | `ch7_adaptation_forest` |

All code is run from the repository root. Inputs live in `outputs/`:
`e1_traces` and `e1_tracking` for §7.2, `e2_chain` and `e2_refstep` for §7.3,
`kappa_loaded` for §7.4, and `final`, `heldout8191` and `aa_floor` for §7.5.

AF arms are *trained* under `PPO_AAM_FD/experiment_scripts`; the §7.4 and §7.5
folders here score them and build the reported tables and figures.

E3 to E6 were moved to `archive/experiments_stale_2026-09-09/` on 2026-09-09:
none feeds a figure or table in the current thesis. They are kept as the record
of questions asked and answered.

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
