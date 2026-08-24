# Stage 0 on the CPU surrogate — what it can and cannot decide

Date: 2026-08-25. Reference: `PPO_RLA_ID_FD/RLA_Experimental_Protocol_Hydraulic_Manipulator.pdf`,
§6 (Stage 0: build the stress region before RLA).

## Summary

The previous sweep in `results/robustness_sweeps.{csv,json}` is **invalid** and must not
be cited. Two independent defects were found and one of them is not fixable on this
surrogate. Stage 0 therefore splits:

| Stage-0 gate | Verdict |
|---|---|
| F1 response authority | **Answerable here — all five axes admitted** |
| F4 observability | **Answerable here — all five axes admitted** |
| F2 control relevance | GPU only (IsaacLab + PhysX) |
| F3 headroom / G1 degradation | GPU only (IsaacLab + PhysX) |

## Defect 1 — the episode set did not move

`frozen_episode_set` required only that a window lie inside the frozen PPO's observation
support. That selected the static bulk of the dataset: the 64 chosen episodes had a
**median cylinder excursion of 0.03 mm over the whole 2.4 s horizon**. A tracking task
the plant satisfies by standing still cannot degrade, which is why the null controller
beat the policy and why the published curves showed error *falling* as stress rose
(authority 0.0 → 3.3 mm; nominal → 8.4 mm).

Fixed: `MIN_MOTION_M = 0.005` in `protocol.py`. 13,050 windows satisfy both support and
motion at H=48, so the constraint costs nothing in sample size. The screen uses only
recorded data and the frozen normalizer, so it stays stress-independent.

## Defect 2 — the surrogate cannot carry an absolute position metric

Open-loop replay (`surrogate_validity.py`, no policy involved: recorded valve commands
replayed through the frozen SD model from a recorded initial state):

| Horizon | Median drift | Median demanded motion | Noise-to-signal |
|---|---|---|---|
| 0.25 s | 2.54 mm | 6.67 mm | 0.37 |
| 0.50 s | 5.45 mm | 12.36 mm | 0.40 |
| 1.00 s | 4.69 mm | 9.56 mm | 0.46 |
| 2.40 s | 7.12 mm | 10.20 mm | 0.62 |

Only 37 of 256 moving windows hold drift below 25% of demanded motion — fewer than the
64 needed. `gate_stage0.py` returns **STOP** on this.

The mechanism is *not* motion insensitivity. The surrogate reproduces motion magnitude
well (excursion ratio 0.97, correlation 0.92 over 1 s; one-step ratio 0.97). Its per-step
*direction* is correlated only 0.75, so integrated position random-walks ≈4.7 mm/s. The
SD model is a usable dynamics model and an unusable position-integrating plant.

Consequence: any metric of the form "absolute tracking error against a reference"
— which is exactly gate G1, *baseline error ≥25% worse than nominal* — is below this
surrogate's noise floor. **G1 must be measured in IsaacLab against PhysX.**

## What survives: F1 and F4

Differential questions do survive, because both arms of a paired point use the same
policy, episodes and initial states, and the stress chain is an exact analytic transform
applied to the valve *before* the surrogate — so the drift is common-mode and largely
cancels. `factor_admission.py`, H=20 (1.0 s), 64 moving in-support episodes, credibility
floor 3.16 mm:

| Axis | First severity clearing the floor | Response authority there | Verdict |
|---|---|---|---|
| delay | 1 tick (50 ms) | 8.19 mm (2.6× floor) | ADMIT — most potent per unit |
| authority | 0.9 (−10%) | 5.69 mm (1.8×) | ADMIT |
| lag | 0.05 s | 5.76 mm (1.8×) | ADMIT |
| deadband | 0.05 | 5.73 mm (1.8×) | ADMIT (0.01–0.02 too weak) |
| rate_limit | 0.2 | 6.02 mm (1.9×) | ADMIT (1.0, 0.5 too weak) |

Observation divergence is 2.4–5.0 normalizer standard deviations at admitted points, so
F4 passes with margin everywhere — plant changes on these axes are plainly visible in
deployable history. Saturation never exceeds 3.6%, so none of these axes works by pinning
the actuator, which is the §12 confound.

## Two cautions on reading the table above

1. **Response authority is not degradation.** A 20 mm trajectory divergence at
   authority 0.1 against ~6.8 mm of demanded motion means the closed loop goes somewhere
   else entirely — it does not mean the task metric got 3× worse. F1 vs F2 is precisely
   this distinction, and F2 is the GPU question.
2. At the harsher severities the observation divergence exceeds 3 standard deviations,
   so the frozen actor is itself extrapolating. Prefer the mildest admitted severity that
   clears F1 when choosing the IsaacLab sweep grid.

## What this changes for the GPU plan

The protocol's §6.1 ordering put valve effectiveness first and additional response
lag/delay last ("only if physically interpretable"). On response authority per unit of
severity, that ordering is inverted: **delay is the strongest axis** — one 50 ms tick
already clears the floor at 2.6×, where authority needs a 10% cut for 1.8×. This matches
the mechanism argument: a factor acting faster than the feedback loop cannot be absorbed
reactively, which is where privileged context has something to buy (§8 oracle gate).

Recommended IsaacLab sweep order: `delay` → `authority` → `lag`, with `deadband` and
`rate_limit` held as alternates. Skip severities below the "first clearing" column.

## Reproduce

```
cd experiments/RMA_PPO/robustness
python3.11 gate_stage0.py          # S0-A surrogate credibility, S0-B nominal control
python3.11 factor_admission.py     # F1 / F4 screens -> stage0_factor_admission.json
```

Both run in about 5 s on CPU. `run_sweeps.py` remains valid *code* but its published
results are superseded; re-running it on this surrogate cannot produce a defensible
degradation envelope for the reason in Defect 2.

---

# Analytic action separation (added 2026-08-25)

`action_separation.py`, 4096 recorded rows, no plant model and no policy rollout.
Full output: `results/action_separation.json`.

Every stress mechanism sits between the commanded and the applied valve, so a
controller that knows the plant can hold the nominal response by commanding the
preimage `g_p^-1(u)`. The gap `du = g_p^-1(u) - u`, mapped through the frozen ID
Jacobian into action units, is a **lower bound on the action separation the §8.2
oracle gate requires**. It also yields analytic saturation — the fraction of rows
where the required command exceeds the ID's `command_limits` — with no surrogate.

## Nominal clipping floor

**5.6% of recorded rows overall, and 21.2% on cylinder 1**, at zero stress. C1's
recorded median |u| is 0.77 against a 0.95 limit: it sits near its command ceiling
before anything is perturbed. All costs below are quoted as excess over this floor.

## Result

| Axis | Separation (frac. of action span) | Extra C1 clipping | Class |
|---|---|---|---|
| **delay** | 0.036 (1 tick) → 0.117 (20) | **+0.2 pp — none** | ANTICIPATION |
| **deadband** | 0.058 (0.01) → 1.13 (0.2) | +1.0 → +29.3 pp | SEPARATION |
| **lag** | 0.044 (0.1 s) → 0.241 (1.0 s) | +7.1 → +22.6 pp | SEPARATION |
| **authority** | 0.073 (0.9) → 0.222 (0.5) | **+11.9 → +69.4 pp** | SEPARATION, costly |
| **rate_limit** | 0.000 at every severity | +0.0 pp | NEGLIGIBLE (static) |

## What this decides

1. **Delay is the axis to run first.** It is the only one that demands a real
   action change while requiring *no additional command authority* — C1 clipping is
   flat to within 0.4 pp across the whole grid. Context has something to buy and the
   actuator is not asked for more, which is exactly the §12 "adaptation-limited"
   region the protocol wants.
2. **Authority stress buys its separation with headroom the machine does not have.**
   A 10% cut already pushes C1 to 33% required-clipping — past the §6.3 30% rule —
   and 0.5 pushes it to 91%. This is the legacy AIF condition (C1 saturated ~92% of
   the scored window) reproduced analytically, before any GPU time. Treat authority
   as §16-B territory: degradation there will be contaminated by saturation.
3. **Deadband is the cheapest strong separation.** d = 0.05 yields 28% of the action
   span for +6.3 pp of C1 clipping. It deserves promotion from "alternate" to a
   primary candidate alongside delay.
4. **Rate limit offers no static separation** — knowing the plant has a rate limit
   does not let you command around it. See the LIMITATIONS note in the module: this
   does not rule out anticipation-type benefit, which the static preimage cannot test.

## Caveat

This is a lower bound on *required* separation, not a prediction of *achieved*
benefit. It establishes that an oracle policy has something distinct to do (a
necessary condition for §8) and where the actuator cannot deliver it. Whether the
task metric actually improves remains gate F2/G1 and remains a PhysX question.
