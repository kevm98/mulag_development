# Pre-registration: the adaptation claim, and the nominal fix

Written 2026-09-06 after the held-out confirmation and the beta ladder, before
either experiment below is run.

## Two gaps this closes

**1. Nothing so far tests ADAPTATION.** Every campaign to date holds the payload
CONSTANT per episode (`MULAG_PAYLOAD_RANGE=M,M`). That measures payload
*conditioning*: the policy is handed a fixed number and behaves better. The
claim the thesis needs is that tracking survives the payload *changing while the
machine drives*, which is a different and stronger statement and has not been
measured.

**2. The nominal-payload harm has an explanation from the literature that none
of my three attempts used.** At 1.0x the online arm loses lateral,
45.99 -> 81.50 mm. Attempts and outcomes:

| attempt | rationale | result |
|---|---|---|
| degree-2 estimator | reduce the 1.41x over-read | 79.65 mm, still significant |
| hard deadband | suppress the estimate near nominal | 89.92 mm, worse |

The deadband was worse for a reason the beta ladder had already established:
gating the OUTPUT makes the context jump discontinuously as the estimate wobbles
across the threshold, and abrupt arrival is what harms this policy. Classical
adaptive control puts the dead zone on the UPDATE LAW, not the output signal.

The remaining explanation is a **train/deploy mismatch**. The policy was trained
on the true, clean, constant context and is deployed on a noisy, ramping
estimate. Under load the adaptation gain outweighs that mismatch; at nominal
there is no gain to be had and only mismatch to suffer. **A-RMA** names the fix:
after the policy and the estimator are trained, unfreeze the policy and
fine-tune it WITH the estimator in the loop, to close the domain gap an
imperfect estimator creates.

## Experiment A: mid-episode payload step

The adaptation test proper. Every environment starts at its base mass and steps
mid-episode:

    MULAG_MASS_STEP_TIME_S=20.0  MULAG_MASS_STEP_JITTER_S=8.0
    MULAG_MASS_STEP_SCALE=3.0    (and 5.0 as a second dose)

`MULAG_MASS_BIMODAL` is deliberately OMITTED. Invariant 21 records that setting
it assigns half the fleet to each mode from t=0, which makes the plant
statically heavier AND the step larger together and confounds the two. Omitted,
the step applies uniformly to every environment and the pre-step baseline is
common to both arms.

The jitter matters: a fixed instant can be compensated on a timer without
sensing anything, so 8 s of jitter is what forces the arm to actually detect the
change.

Arms, all the SAME checkpoint:

* `ctxzero` -- context zeroed, which IS the frozen policy bit for bit
* `ctxest` -- payload estimated online, `beta = 0.02`
* `ctxest_b0005` -- the same at `beta = 0.005`, the ladder's best rung

**Primary:** `ctxzero -> ctxest`, paired per environment, under the Table 7.7
rule. **Secondary:** contact, hard impacts, tool acceleration, safety.

A pass here is the claim the thesis wants and none of the constant-payload
results can support: the arm tracks a payload that changes under it.

## Experiment B: A-RMA phase 3

Retrain the same 1024 context parameters with `MULAG_CONTEXT_ABLATE=estimated`
and the fitted estimator in the loop, so the policy is trained on the
distribution it is deployed on. Everything else identical to run 1.

Scored at 1.0x, 3.0x and 5.0x against `ctxzero`.

**Declared in advance:**

* **Success** is the nominal harm removed -- `ctxzero -> ctxest` at 1.0x no
  longer significantly negative on lateral -- **while 3.0x and 5.0x still pass**.
  A fix that rescues nominal by destroying the loaded gain is not a fix, and
  both are scored to make that visible.
* If nominal is fixed and the loaded gains shrink but still pass, that is
  reported as a trade with both numbers quoted.
* If nominal is NOT fixed, this is the fourth failed attempt and I stop
  patching. The claim is then scoped explicitly to loaded operation, stated as
  "gate the adaptation off below a known payload threshold", and the nominal
  behaviour is reported as a limitation rather than buried.

## What is NOT re-decided

The statistic, the bounds and the arm definitions are unchanged. The held-out
result at `beta = 0.02` stands as measured; nothing here restates it. Seed
replication is deferred until the configuration is final, so that seeds validate
the arm that will actually be claimed rather than a superseded one.
