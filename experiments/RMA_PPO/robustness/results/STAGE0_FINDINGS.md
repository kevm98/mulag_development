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
