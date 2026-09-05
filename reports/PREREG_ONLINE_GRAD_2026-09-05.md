# Pre-registration: online gradient adaptation of the policy

Written 2026-09-05, before any arm is implemented or run. This is a new
mechanism, not a repair of the residual campaign.

## What is being tested

Every adaptation mechanism in this thesis so far is trained offline and frozen at
deployment. Nothing changes at runtime. This tests the literal reading of the
title: **policy parameters updated by gradient descent while the machine drives.**

## The signal

The deployed stack computes, every control step, the forward model's prediction
of the cylinder increment and the increment PhysX actually realised. Their
difference is the actuator prediction residual, already carried in observation
channels 58:62. It is a direct, dense, per-step measurement of how far the plant
has drifted from the model the policy was trained against. No privileged
information is used.

Call it `d`. If the plant delivers more travel than the model predicts, `d > 0`
and the policy should command less.

## The mechanism

Let `u0 = pi_frozen(o)` and `u = pi_theta(o)`, where `theta` is a small trainable
adapter initialised so that `u = u0` exactly. At each step:

    d_hat  <- (1 - beta) * d_hat + beta * d              (EMA of the measured residual)
    target <- f(x, u0) - d_hat                           (what the command must ask for)
    L      <- || f(x, u) - target ||^2
    theta  <- theta - eta * dL/dtheta                    (one step, per control tick)

`f` is the frozen forward model, differentiable in `u`, condition number 2.14 on
the extension Jacobian (Section 5.3.1). `theta` is the actor's final layer only.

Adaptation is online, per step, and resets to the frozen weights at every episode
boundary. No reward, no rollouts, no credit assignment.

## Guards, fixed here

- `theta` projected to a bounded neighbourhood of the frozen weights each step.
- Adaptation frozen when `|u| < 0.05` on an axis, because a quiet command makes
  the mismatch unidentifiable.
- Adaptation frozen during contact transients flagged by the environment.
- Hard revert to frozen weights if scored error over a trailing window exceeds
  the frozen base by more than 50 %.
- Every guard is logged with the fraction of steps it fired.

## Arms

| Arm | Adaptation | Purpose |
|---|---|---|
| `off` | none, frozen base | the existing baseline |
| `on` | as above | the mechanism |
| `sham` | same update, gradient sign randomised per episode | same compute and parameter motion, no valid direction |

The `sham` arm is the control the residual campaign taught us to include from the
start. If `on` beats `off` and `sham` does not, the direction of the gradient is
doing the work. If `sham` also beats `off`, the benefit is parameter motion
rather than adaptation, and the result is reported as such.

## Fixed before the first run

- **Primary:** `off` -> `on`, paired per environment.
- **Control:** `off` -> `sham`. Reported always, not only if it is favourable.
- **Success rule:** identical to Table 7.7. One channel improves with its paired
  bootstrap interval excluding zero, and the other improves or stays within its
  non-inferiority bound (cut height 10 %, lateral 1.5 %). Bounds are the measured
  A/A floor from `PREREG_R2_2026-09-01.md` and are not revisited.
- **Statistic:** paired per-environment median difference, percentile bootstrap
  B = 20000, percentage as the median log-ratio.
- **Evaluation:** the same three reserved meshes 2718, 3141, 8191, 176
  environments, 900 steps, same plant as the starved campaign
  (`MULAG_PAYLOAD_RANGE=1.0,5.0`, resampled per episode).
- **Verdict:** positive only if `on` passes against `off` **and** `sham` does not.

## Learning rate

`eta` is a genuine hyperparameter and will be reported as one. It is selected on
mesh 2718 alone, before the scored evaluation, from the fixed ladder
{1e-5, 1e-4, 1e-3}, on training-side prediction loss only, never on the scored
gate metrics. The selected value and the ladder are reported.

## Declared outcomes

- **Positive:** `on` passes, `sham` does not. Online gradient adaptation improves
  closed-loop tracking on this machine.
- **Negative:** `on` fails, or `sham` passes too. Reported as such. The residual
  and latent results stand unchanged either way; this is an additional mechanism,
  not a replacement for them.

## Stopping

If the mechanism is unstable at every rung of the learning-rate ladder, that is
the result and it is reported. No further variants, no reward shaping, no widening
of the adapter.
