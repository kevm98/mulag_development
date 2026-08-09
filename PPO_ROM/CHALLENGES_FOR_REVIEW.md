# PPO3_1 — current challenges, for external review

**PLEASE DO NOT CHANGE ANY CODE. Advice, diagnosis and calculation only.**
We want your reasoning and numbers; we will make the edits ourselves.

---

## 1. System

Hydraulic manipulator (MULAG) on a driving truck, verge-mowing task, Isaac Lab +
PhysX, `rsl_rl` PPO.

* Action space: **4** — hydraulic cylinder velocity commands c1..c4. c4 is the
  head-pitch cylinder. Observation width 66.
* Cylinder targets are integrated and clamped by an anti-windup lead cap:
  `(0.018, 0.018, 0.018, 0.30)` m; action scale `(0.004, 0.004, 0.004, 0.012)` m.
  c4 deliberately has ~16.7x the lead and 3x the action scale (it is the
  corrective axis for head pitch).
* c4 is **not** force-limited: measured 6.8 % of its 320 kN effort limit,
  saturating 0.026 % of steps. c1 is the loaded axis (53 %, 1.3 % saturation).
* Task: keep the cutter on a tracked line on a verge, at target cut height,
  head pointing down, in ground contact.
* Terrain: procedurally generated, stage 6 = flat ground with discrete raised
  chunks, lateral path amplitude scale 2.2, 11 path families.

## 2. Training setup

Two stages, teacher-student:

1. **Distillation.** A privileged DLS (damped least squares) teacher is
   distilled into the student by behaviour cloning.
   Teacher measured on this plant: lateral error 0.122 m, cut-height error
   0.023 m, p01 head alignment 0.964, ground contact 0.781, p95 tool
   acceleration 64.9 m/s^2. Student checkpoint written with std = 0.10 on all
   4 axes.
2. **PPO refine** from that student, with the critic head reset (the distilled
   critic is untrained).

PPO hyperparameters (rsl_rl):

```
learning_rate      1.0e-3, schedule = "adaptive" (KL-adaptive)
entropy_coef       0.005
init_std           0.30      (not applied on resume; std comes from checkpoint)
num_steps_per_env  128
num_envs           256
max_iterations     1500-3000
```

## 3. The reward (9 bounded terms) — we believe this part is CORRECT

Two-scale shapes, no Gaussians, every term bounded and logged individually.

```
term                weight    shape
cut_height           +6.0     Huber, scale 0.04 m, cap 2.5
lateral             +10.0     two-scale linear: fine 0.15 m (40 %),
                              coarse 1.5 m (60 %)
ground_contact       +4.0     linear in contact fraction
head_alignment      -12.0     two-scale: fine 0.02, coarse 1.0
                              below threshold 0.98
contact_loss         -6.0     linear
action_rate          -1.0     rate, NOT magnitude
tool_acceleration    -1.0     scale 300 m/s^2
contact_force        -1.0     above 30 kN, scale 150 kN
cylinder_headroom    -2.0     barrier, margin 0.02 m, scale 0.1 m
task total  +20.0     safety total  -23.0
```

Verified by autograd across the whole operating range — **no dead gradients**:

```
lateral  |dR/de| :  30.7 @ 0.05 m,  4.0 @ 0.30 m,  4.0 @ 0.83 m,  4.0 @ 1.40 m
head     |dR/da| :   6.0 @ 0.96,    6.0 @ 0.90,    6.0 @ 0.666,   6.0 @ 0.24
```

Exchange rates: 1.00 of contact fraction = 10.0 reward units; 1.00 m of lateral
error = 4.0 reward units (coarse slope).

## 4. The problem

We scored three policies under this exact reward:

| policy   | reward  | lateral | head p01 | contact | cut height |
|----------|---------|---------|----------|---------|------------|
| teacher  | **+13.44** | 0.122 m | 0.985 | 0.781 | 0.023 m |
| null (zero action) | **+12.26** | 0.175 m | 0.990 | 0.721 | 0.013 m |
| PPO @ iter 850 | **+3.24** | 0.832 m | 0.666 | 0.880 | 0.015 m |

**PPO trained for 850 iterations from the +13.44 student and ended at +3.24 —
a quarter of the value of doing nothing.** The reward ranks all three
correctly, so the reward is not paying for the wrong behaviour. The policy
descended its own objective.

The trade PPO made is explicitly penalised by the reward:
contact +0.159 = **+1.59**, lateral +0.657 m = **-2.63**. Net negative.

### The measurement we think explains it

Exploration std (`distribution.std_param`, 4 axes) over that run:

```
student   0.100  0.100  0.100  0.100
iter 200  0.140  0.158  0.121  0.140
iter 400  0.187  0.207  0.144  0.161
iter 850  0.310  0.290  0.339  0.237
```

Std grew ~3x monotonically. This repo has a prior independent measurement:
setting exploration std to **0.35** on all axes flipped head alignment from
0.9507 to **0.5852**. At iter 850 std is ~0.31 and head alignment is **0.666**.

Hypothesis: `entropy_coef = 0.005` inflates the policy std into a regime where
the injected action noise mechanically flips the head; the -12.0 head penalty
then dominates and the policy never recovers. Note the *previous* failure mode
in this project was the opposite — std frozen byte-identical across checkpoints
(~0.4 mm of exploration per step) with `entropy_coef = 0.001` and
`learning_rate = 5e-7 fixed`. The fix appears to have overshot.

Complication we are unsure how to reason about: **rsl_rl normalizes advantages**,
so the reward's absolute scale is divided out of the surrogate objective. A
9-unit drop in return therefore does not straightforwardly compete with a fixed
entropy coefficient. We do not have a principled way to set `entropy_coef`
against a reward of this magnitude.

## 5. What we would like from you

**Diagnosis**

1. Is the entropy-inflation hypothesis the right read of the std trajectory and
   the head collapse? What else would produce a monotone 3x std growth over 850
   iterations in PPO with a KL-adaptive learning rate?
2. Is a KL-adaptive LR schedule at 1e-3 appropriate when starting from a
   behaviour-cloned warm start, or does it permit the policy to walk away from
   the student before the reset critic has a usable value estimate?
3. Is resetting the critic head correct here, or does the resulting garbage
   advantage estimate for the first N iterations itself cause the descent? If
   so, what is the standard remedy — critic warm-up with a frozen actor,
   or something else?

**Calculations**

4. Given the measured trajectory (std 0.10 -> 0.31 over 850 iters,
   `entropy_coef = 0.005`, 4 action dimensions, 256 envs x 128 steps per
   iteration, KL-adaptive LR at 1e-3): estimate the `entropy_coef` that would
   hold std inside a target band of roughly [0.08, 0.15]. Please show how you
   account for advantage normalization.
5. Head alignment is measured to flip at std ~0.30-0.35 applied uniformly.
   Given c4 has 16.7x the lead cap and 3x the action scale of c1-c3, compute a
   defensible **per-axis** std cap. Should the c4 std be bounded separately, and
   at what value?
6. Compare against Isaac Lab / rsl_rl locomotion defaults: what do the standard
   configs use for `entropy_coef`, `init_std`, and do they bound log_std from
   above? Is an explicit std ceiling or entropy annealing the conventional fix?
7. If we add a KL-to-reference regulariser to keep PPO near the distilled
   student, what coefficient would you recommend, and how should it be annealed?

**Acceptance criteria**

8. Our worst-family ground-contact gate is set at 0.850. Measured: null policy
   0.635, PPO@850 0.758, and no policy in this project's history has exceeded
   ~0.75 worst-family. The privileged teacher reaches only 0.781 *mean*. Is 0.850
   defensible, or is it a gate nothing can pass? How would you re-derive it from
   the achievable band?
9. More generally: the null (zero-action) policy passes 12 of our 17 gates and
   scores 7.74. How should acceptance criteria be constructed so that a
   do-nothing policy fails clearly — i.e. so the gates measure the task rather
   than the terrain?

## 6. Known open issue, separate from the above

`PPO_ROM/scripts/evaluate.py` scores the **old shared 17-gate**
contract (`summarize_convergence`), while the rebuilt 8-gate multi-mesh contract
in `PPO_ROM/gates.py` is currently referenced only by unit tests and
the preflight script. The acceptance path and the intended acceptance contract
are not the same thing. We are aware; flagged here for completeness.

**Update 2026-08-06: this reached further than evaluate.py.** `train.py
--until_converged` used the same legacy contract as its STOP CONDITION, via
`convergence_rollout` -> `summarize_convergence`. Three of those thresholds are
above anything this plant has produced, the privileged DLS teacher included:
`ground_contact` 0.90 and `worst_family_contact` 0.85 against a teacher
measuring 0.7514-0.8132, and `hard_impacts` as a raw EVENT COUNT bounded at 0.
The flag could therefore never fire: `model_converged.pt` would never be
written and the run would quietly train to its iteration ceiling while
appearing to have a convergence criterion. `train.py` now refuses the flag with
that explanation. Acceptance runs through `scripts/evaluate_gates.py`, which is
the only entry point implementing the 8-gate multi-mesh contract.

`convergence_rollout` cannot be repointed at `gates.py` as-is: `check_gates`
requires >= 3 distinct terrain meshes and the in-training rollout has one env
build. Making the stop condition honest means either rebuilding the env per
mesh mid-training or accepting a fixed iteration budget; the fixed budget was
chosen.

## 7. Resolution so far (2026-08-06) — evidence-based gate re-derivation

The student 3-mesh evaluation (meshes 1337/17/909, 32 envs x 480 steps per
mesh, `reports/frontier/student_gates.json`) and the DLS teacher evaluation on
the same meshes (`reports/frontier/teacher_gates.json`) produced a decisive
measurement: **the privileged teacher fails the same four gates the student
fails, and the null (do-nothing) policy passes the same four both pass.** The
contract could not distinguish doing nothing from a ground-truth controller,
so the thresholds, not the policies, were the defect.

Three thresholds were re-derived from the achievable band measured on the
teacher, and verified against every measured policy (pure Python, no
simulator):

* **ground_contact 0.850 -> 0.700.** Teacher worst-family: 0.8132/0.7514/0.7694;
  its mean is 0.781; null is 0.635. 0.850 was above everything this system has
  ever produced. At 0.700 the teacher passes (+0.0514), the student passes
  (+0.0076, thin but real) and null fails (0.635).
* **hard_impacts 0.020 -> 0.100.** Teacher worst mesh: 0.0870 (4.4x over the old
  gate); the student is BETTER (0.0678). 0.020 was unreachable by the
  ground-truth controller. At 0.100 both pass.
* **safety_terminations 0.050 -> 0.080.** Teacher worst mesh 0.0674, student
  0.0611, null 0.0025. At 0.080 both teacher and student pass with margin.

**Lateral (0.300 worst-family) is NOT re-derived yet, and that is deliberate.**
The family breakdown shows exactly two families drive it: boom_sweep and
gaussian_slalom, both over 0.300 for the teacher as well (boom_sweep
0.3253/0.3657, gaussian_slalom 0.3528 on m1337).

**The lead-cap evaluation has now run, and it FALSIFIED the premise.** The
repo's help text claims relaxing c1-c3 0.018 -> 0.050 halves boom_sweep lateral
error (0.3745 -> 0.1602) with zero retraining. Measured on all three meshes at
32 envs x 480 steps, the same relaxation moves boom_sweep the *other* way:

```
mesh    boom_sweep 0.018    boom_sweep 0.050
1337          0.2800              0.4849
17            0.3189              0.3281
909           0.4046              0.5134
```

Safety collapsed with it (worst mesh): p01 head alignment 0.9597 -> 0.8814
(gate FAIL), hard-impact fraction 0.0678 -> 0.2698, worst-family contact
0.7076 -> 0.4722, and **min cylinder headroom 0.0387 -> 0.0000 exactly on all
three meshes** -- the arm bottoms out against its own travel limits. Failing
gates went 3 -> 6 of 8. This happened with c4 held at 0.30, so c1-c3 alone are
sufficient to break the head gate.

The family pattern inverts the hypothesis: all three families that require a
*sustained* offset degraded (boom_sweep 0.2800 -> 0.4849, lane_change_hold
0.1102 -> 0.1968, offset_return 0.0759 -> 0.1633), while the one fast-reversal
family improved (gaussian_slalom 0.2818 -> 0.1405). Reading: relaxing an
anti-windup clamp reintroduces windup, which harms held commands and is
irrelevant to commands that reverse quickly. So 0.018 appears correctly tuned.

Authority spent without tracking improvement, terminating at zero headroom, is
the signature of a target beyond kinematic reach rather than a command clamp
holding back a capable arm. Full write-up:
`reports/frontier/LEAD_CAP_EXPERIMENT.md`.

**Question 10 for review:** given the teacher also fails boom_sweep and
gaussian_slalom on every mesh, and extra command authority makes tracking worse
while driving the arm to its stroke limits, is it defensible to conclude these
two path families are kinematically infeasible at amplitude scale 2.2 -- and to
re-derive the lateral gate excluding them (or reduce their amplitude) rather
than continue tuning the plant? What measurement would you require before
accepting an "infeasible by construction" verdict?

---

Again: **please do not modify any code.** Reasoning, numbers and
recommendations only.
