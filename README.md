# MULAG verge mower — learned hydraulics and online adaptation

A MULAG FME 500 verge mower on a Unimog carrier. The truck drives the road at a
constant speed; the arm reaches over the embankment and the cutter follows a
line along the verge while staying on the ground. Tracking is the objective and
the terrain is a disturbance, not the task.

This repository holds the reinforcement-learning pipelines, the learned actuator
models they act through, the experiments, and the thesis that reports them:
*Learning-Based Online Adaptation for Control of a Hydraulic Manipulator*.

All evidence here is simulation-to-simulation. Simscape supplies the actuator
data, IsaacLab is the deployment simulator, and the deployment plant is itself a
learned model. Nothing here is a claim about hardware.

---

## The three control stacks

Each stack is a complete controller-and-actuator pair, not an actuator that can
be swapped and re-scored. The thesis names them R, IF and AF.

| | stack | command path | what it carries |
|---|---|---|---|
| **R** | [PPO_ROM](PPO_ROM/) | policy → command ROM → PhysX | reference condition. The ROM plus PhysX *is* the deployment plant, so R defines the target rather than competing for it. |
| **IF** | [PPO_ID_FD](PPO_ID_FD/) | policy → inverse model → forward model → PhysX | the non-adaptive learned baseline. Matched to R on every recorded training field. |
| **AF** | [PPO_AAM_FD](PPO_AAM_FD/) | policy → valve command → bounded history-conditioned residual → forward model → PhysX | the adaptive stack. Carries the online-adaptation result. |

R and IF are the matched pair: they agree on seed, terrain seed, environments,
curriculum stage, learning rate, entropy coefficient, normaliser and warm-up, and
differ only in the actuator path. A difference between them is attributable to
the representation.

**AF is not a third column.** Its policy emits the valve command itself where the
other two emit a bounded cylinder increment, so the quantity being chosen is not
the same quantity. Its claims are made within the stack, against its own frozen
base and a capacity-matched control. Cross-stack numbers scored on AF's earlier
inverse-model architecture are withdrawn, not reported.

### Where each stack stands

| | result | limit |
|---|---|---|
| R | median lateral error 0.0392 m over 33 held-out (mesh, family) cells | fails the lateral gate on its worst family, worst mesh |
| IF | median lateral error 0.0281 m, lower than R on the pooled mean | fails head alignment at 0.2247, the fine-tuning collapse |
| AF | +62.6 % cut and +78.6 % lateral against its frozen base on the best seed, 880 paired environments | holds on two of five training seeds; the median seed does not meet the pre-registered bar |

Policy-only control through learned hydraulics is demonstrated on both
representations, with the classical controller absent from the deployed loop. It
is demonstrated, not certified: neither stack satisfies the whole eight-gate
acceptance contract.

---

## Three findings that govern everything here

**Supervised accuracy is not evidence of control authority.** The first forward
model scored R² ≈ 0.999 while learning to ignore the valve command and
reconstruct motion from the state history. Validation loss did not show it;
ablating the command did. Excitation design paired with input ablation is the
test that exposes it, and it is required for any learned actuator model intended
for control rather than prediction.

**The task has to be inside the plant's capability.** The cylinders move the
cutter sideways at 0.240 m/s at most. Under the old settings most path families
demanded more than that, so error accumulated by construction and even the
privileged DLS teacher failed. No reward, critic or PPO change fixes a task the
plant cannot execute. Terrain is now derived from that ceiling.

**On a saturating actuator, placement beats estimation.** A bounded correction on
the inverse model's output produced no closed-loop effect, because that command
already rested on its clip for most of the scored window. The same information
delivered onto the valve command converts into a large gain. The binding
constraint was delivery, not the quality of the estimate.

---

## Layout

### Pipelines

    PPO_ROM/          R. Command ROM actuator.
    PPO_ID_FD/        IF. Inverse and forward models in series.
    PPO_AAM_FD/       AF. Direct valve command with a history-conditioned residual.
    PPO_ID_AAM_FD/    the retired latent-conditioned AF. Provenance only.

### Shared libraries

Imported by the pipelines, not pipelines themselves. `verge_task`'s task class
inherits from `mulag_plant.policy.env`, and every pipeline imports both.

    verge_task/       terrain generator, observation contract, DLS teacher,
                      distillation and training scripts, actuator resolution
    mulag_plant/      base environment, DLS solver, ROM runtime, cylinder adapters

### Actuator models

Each folder holds the weights *and* the model definitions, because the runtimes
import the definitions from beside the weights.

    rom_model/        the calibrated command ROM              → PPO_ROM
    id_fd_model/      inverse and forward models              → PPO_ID_FD, PPO_AAM_FD
    sd_model/         state-dynamics model. Not in the thesis.
    outputs/stage31_rom/   AAM and latent-adapter checkpoints → PPO_AAM_FD

Read each folder's README before touching it. The trailing space in two
`id_fd_model/vendor_models` directory names is real, and the *strict* forward
model is the validated one; the control variant was rank ~2 and under-fit the
valve.

### Experiments, data and the thesis

    experiments/      E1 to E6, one folder each. See experiments/README.md for
                      what each one answers and whether the thesis cites it.
    AAM_module/       dataset builders and the frozen Simscape and IsaacLab
                      collections. The _frozen copies are the ones results use.
    outputs/rif_delay/  per-environment scored reports behind Chapter 7.
    logs/rsl_rl/      training runs and checkpoints.
    reports/          pre-registrations, written before each campaign.
    thesis/           thesis sources. Build with pdflatex from
                      thesis/thesis/hydraulic_manipulator_thesis.
    videos/           rollout recordings.
    archive/          retired pipelines and superseded material. Provenance only.

Every artifact whose SHA-256 the thesis pins in Appendix C is tracked in git.

---

## Common commands

Each pipeline README is the command reference. These are the entry points.

```bash
# watch the shipped policy
PPO_ROM/visualize.sh

# preflight before committing GPU hours
PPO_ROM/scripts/preflight.py --num_envs 32 --steps 120
PPO_AAM_FD/scripts/preflight.py --num-envs 4 --device cpu

# score on the eight gates, one process per mesh
MESHES="1337 17 909 4242 555 2024" \
  PPO_ROM/scripts/score_per_mesh.sh <name> <ckpt.pt> 480 32

# train and score an adaptation arm
PPO_AAM_FD/scripts/train_residual_adapter.py <frozen_base.pt> \
  --num_envs 256 --iterations 400 --history 20 --alpha 0.5 --seed 17
PPO_AAM_FD/scripts/evaluate_gates.py <frozen_base.pt> \
  --residual_checkpoint <arm>/model_399.pt --residual_alpha 0.5


---


