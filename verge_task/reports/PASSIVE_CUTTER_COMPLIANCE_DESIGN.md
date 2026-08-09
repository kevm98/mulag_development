# Design note — passive floating cutter suspension

**Status: DESIGN ONLY. Not implemented, not enabled, not measured.**

This is the Phase 6 fallback. It must not be built until Stage B and Stage C
have run to completion and their strict gates have been read, because if
retraining on the corrected plant succeeds then a mechanism change is an
unnecessary and unfalsifiable addition to the plant.

Prepared now so that the decision, when it comes, is made from a written
design rather than improvised under pressure.

---

## 1. When this becomes the right answer

Only if, after Stage B and Stage C on the frozen plant:

* PPO3 or PPO4 still produces `visible_lift_event_count > 0` in any strict
  cell, **and**
* the remaining lifts are still impact/ejection events rather than tracking
  failures (i.e. separation follows a force peak with upward velocity), **and**
* further reward-only tuning has been stopped.

If the residual lifts are *tracking* failures — the cutter descending too
slowly into a hollow, no force spike, no upward velocity — a suspension will
not help and the problem is the policy's descent rate, not impact energy.
**Classify the residual events with `probe_contact_loss.py` before deciding.**

## 2. Why this is not a second controller

The "PPO is the sole controller" requirement forbids a control law that
observes state and computes a command underneath the policy. A passive
spring-damper is not that: it has no setpoint, no gain on an error signal, no
access to any measurement, and no command output. It is a mechanical element
with a force that is a fixed function of its own deflection and deflection
rate.

The distinction that matters operationally: **the policy's action still maps
to the same cylinder targets, and nothing writes a target the policy did not
ask for.** The existing structural test
(`test_the_state_machine_writes_no_plant_command`) already enforces that for
the contact machinery; the equivalent for this would be that no new code path
writes `_cylinder_pos_target`, `_joint_ref`, `_current_valve_cmd` or
`_fd_delta_corrected`.

Real verge mowers carry exactly this: a floating head on a sprung linkage.
Adding it makes the simulated plant *more* like the machine, not less.

## 3. Proposed mechanism

A single prismatic degree of freedom between the tool mount and the cutter
body, along the tool's own vertical axis.

| property | proposed | rationale |
|---|---|---|
| travel | ±40 mm, hard-stopped | The measured airborne gaps are 0.24–1.18 m, far beyond any suspension; the point is to absorb the impulse, not to follow terrain. 40 mm is ~8× the 5 mm strict lift tolerance. |
| stiffness | 2e5 N/m initial | At the measured 15–20 kN mean contact force this deflects 75–100 mm — too soft. Start instead from *force at full travel*: 40 mm × k = the force we are willing to transmit. For 20 kN that is k = 5e5 N/m. **Sweep 3e5–8e5.** |
| damping | critically damped or above | The V1 finding that stiffness alone returns stored energy applies here too. With an effective cutter mass m, c = 2·sqrt(k·m); measure m from the asset rather than assuming. Under-damping recreates the ejection in a new place. |
| preload | gravity-balanced | The head must rest at mid-travel under its own weight, or the suspension consumes its stroke statically and offers nothing on impact. |

Implementation would be an IsaacLab prismatic joint with an implicit actuator
at the chosen stiffness/damping and zero target — **not** a PD controller with
a computed setpoint. The distinction is that the target never changes.

## 4. The one measurement that decides the parameters

`terrain_compliant_contact_stiffness` already exists and is **inert** — a 10×
sweep moved median peak force 50.4 → 46.9 kN, i.e. PhysX does not honour it
for this contact pair. That is the reason to put the compliance in a *joint*
rather than in the contact material, and it is also a warning: **verify the
joint actually deflects before believing any downstream number.** A zero-action
rollout logging the prismatic joint position against contact force answers
that in one run, and must be the first thing done.

## 5. Required tests before any result is reported

Per the instruction, with zero policy action and with both actuators:

1. **It absorbs impact energy.** Peak contact force falls materially against
   the same rollout without the joint. Compare on `boom_sweep @ 1.0`, the
   measured worst cell.
2. **It prevents upward ejection.** `impact_ejection_event_count` and
   `maximum_positive_cutter_vertical_velocity_mps` both fall. The V1 pre-fix
   figures are 4.41 m/s (PPO3) and 7.93 m/s (PPO4).
3. **It does not destabilise the truck.** `p95_chassis_lateral_reaction_mm`
   and `p95_chassis_attitude_reaction_deg` stay inside their gates (25 mm,
   0.30°). A sprung mass on the end of the boom is a new oscillator and could
   couple into the chassis.
4. **It preserves cutter orientation.** `min_head_alignment` and
   `p01_head_alignment` do not regress. A single prismatic DOF should not
   rotate the head, but the joint frame must be verified against the head-down
   axis, not assumed.
5. **It does not create unlimited ground conformance.** The hard stops must be
   reached in the measured terrain — if the cutter never uses its travel the
   joint is inert, and if it never bottoms out the travel is unrealistically
   large. Log the deflection distribution and require both tails to be
   populated.
6. **Zero-action baseline first.** Every one of the above compared against the
   same configuration with the joint locked, so the joint is the only variable.

## 6. What this would invalidate

Every stored PPO3/PPO4 number was measured on a rigid tool mount. A suspension
changes the plant, so:

* the historical 17 gates would need re-baselining, and a score from before
  the change is not comparable with one after;
* both checkpoints would need retraining, not fine-tuning — the cutter's
  vertical dynamics change qualitatively;
* `cutter_contact_reference_height_m` (−0.30) is calibrated to the rigid
  geometry and would have to be re-measured, because the cutter's resting
  height relative to the head origin becomes load-dependent.

That cost is the reason this is a fallback and not the first move.

## 7. Branch

To be created as `design/passive-cutter-compliance` off the current work, so
the mechanism change never lands on the same branch as the policy result and
the two can be evaluated independently.
