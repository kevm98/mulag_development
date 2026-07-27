# PPO2 ROM-B remediation: root-cause report

Checkpoint under audit
`logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt`
SHA-256 `6731bac46e083d7ac99074e3a7b34752b64117a5003755e3d8296747353a69e9`

Evidence base: `PPO2_ROM_B_PhysX/reports/evaluation_2026-07-24_19-28-48/rom_b_held_out_report.json`
(25 Monte Carlo rows + 135 h-sweep rows, 128 environments and 120 scored steps each),
plus the curriculum preflight artifacts under `logs/rsl_rl/ppo2_rom_b_robust/`.

All numbers below are recomputed from those artifacts. No new rollout was run.

---

## 1. Summary

The failures are **not** ROM-B robustness failures. Four independent findings:

1. **The failures are h-independent.** Pooled termination is 1.5–2.0 % everywhere,
   and which `h` values fail is sampling noise, not degradation.
2. **The failures pre-date ROM-B.** At `randomization_scale = 0.0` (pure nominal
   ROM-A), the same policy family already terminates on `line` with clearance
   causes and on `square` with head-alignment causes, at the same magnitudes.
3. **Three of the seven promotion gates cannot fail**, because they are numerically
   identical to the environment's termination thresholds (or are clamped by
   construction). All acceptance pressure therefore lands on the single
   termination gate, and every clearance/head violation is re-expressed as a
   termination.
4. **The clearance and head-alignment shaping has no gradient over the operating
   range.** Both penalties switch on at 90–95 % of the hard limit, so the policy's
   equilibrium sits against the limit rather than away from it.

---

## 2. Configuration parity: evaluator vs training

Checked line by line between `scripts/train.py` and `scripts/evaluate.py`.

| Item | Training | Evaluation | Parity |
|---|---|---|---|
| Residual authority | `apply_residual_curriculum_stage(cfg, 4)`, `residual_scale_m=0.002`, `cylinder4_residual_scale_m=0.00025` | identical | matches |
| Path speed | `0.18` | `0.18` | matches |
| Trajectory library | `plant_reachable_trajectory_library_v3_large.json` | identical | matches |
| Shape randomization | `randomize_trajectory_shape=False`, `procedural_trajectory_probability=0.0` | identical | matches |
| Clearance reference | `clearance_relative_to_reference=True`, band `0.20` (set by stage 4) | identical | matches |
| Safety thresholds | head `0.98`, cylinder margin `0.002`, truck post `0.02` / raw `0.05` | identical | matches |
| Actor inference | `runner.get_inference_policy` (deterministic mean) | identical | matches |
| Warmup exclusion | `_preposition_counter <= 0 & _settle_counter <= 0` | identical | matches |
| ROM-B factor ranges | `mode="random"`, scale `1.0` at stage 4 | identical | matches |
| **Safety terminations** | **`defer_safety_terminations = True`** | **probe forces `False`** | **differs by design** |
| **Reset synchronization** | **`synchronize_resets = True`** | **probe forces `False`** | **differs by design** |

The clearance signal is consistent across shaping, ROM-B shaping and termination
(`PPO2_ROM_PhysX/policy/env.py:1030`, `:1062`, `:1212` all subtract
`_original_reference_world_z`, which is refreshed every step at `:877`). No
stale-reference bug.

### 2.1 The one real mismatch

During PPO collection, `defer_safety_terminations=True` reduces
`_get_dones` to `terminated = nonfinite` (`policy/env.py:1229-1232`). Crossing
clearance `0.20 m`, head `0.980`, a cylinder limit, or a truck-drift limit
therefore costs the policy a **bounded per-step reward penalty** and nothing else —
the episode continues.

Under the evaluation probe the same crossing **ends the episode and is counted as a
termination**. The policy has consequently never been trained against the cost
structure it is graded on. This is deliberate (the comment at `:1222-1228`
explains that synchronized resets used to stall training), but it is the reason
the policy has no incentive to hold a margin: it learned that a violation is a
recoverable few-step penalty.

---

## 3. The failures are h-independent

Pooled over the entire h-sweep (9 h × 3 rounds × 128 envs = 3456 per shape):

| shape | pooled termination | dominant cause |
|---|---:|---|
| line | 130/3456 = **0.0376** | clearance 96, truck_raw_reaction 34 |
| square | 83/3456 = 0.0240 | head_alignment 50, truck_raw_reaction 32 |
| arc | 66/3456 = 0.0191 | head_alignment 34, truck_raw_reaction 32 |
| sine | 39/3456 = 0.0113 | truck_raw_reaction 32 |
| circle | 26/3456 = 0.0075 | truck_raw_reaction 26 |
| **all** | 344/17280 = **0.0199** | |

Monte Carlo (5 × 5 × 128) gives the same ordering: line 0.0344, square 0.0219,
arc 0.0172, sine 0.0016, circle 0.0000; overall **0.0150**.

The gate is `max` over rows of a 128-sample binomial rate, compared against `0.05`.
A row fails when `k >= 7`. For a true per-row rate `p`:

| true `p` | P(one row fails) | P(any of 25 MC rows) | P(any of 15 rows at one `h`) |
|---:|---:|---:|---:|
| 0.010 | 0.0003 | 0.008 | 0.005 |
| 0.020 | 0.0149 | 0.313 | 0.202 |
| 0.030 | 0.0915 | 0.909 | 0.763 |
| 0.037 (line) | ~0.20 | ~0.67 (5 line rows) | ~0.49 (3 line rows) |

With line at `p ≈ 0.037`, at least one line row exceeds the gate at a given `h`
with probability ≈ 0.49, so across nine `h` values the sweep reports failures with
probability ≈ 0.998, and **which** `h` values fail is arbitrary. That is exactly the
observed pattern: 0.4 FAIL, 0.5 PASS, 0.6 FAIL, 0.7/0.8 PASS, 0.9 FAIL, 1.0 PASS,
1.1/1.2 FAIL — physically incoherent, statistically expected.

Two structural amplifiers:

* The h-sweep reuses the **same 15 seed groups at every `h`** (`evaluate.py:422`).
  Seed 400027 (`line`, round 3) is chronically bad: 0.078 at h=0.4, 0.070 at h=0.9,
  0.055 at h=1.1, 0.047 at h=0.7. The sweep re-samples the same adversarial initial
  conditions nine times and reports the resulting correlated failures as an h effect.
* The reported "acceleration" failures are the same story. Both `p95_acceleration`
  breaches (h=0.6 → 40.05, h=0.9 → 42.21) come from **one** row, `circle` seed
  400030, whose acc95 across the nine h values is
  35.8, 37.8, **40.1**, 34.9, 38.0, **42.2**, 33.1, 37.9, 35.5 — non-monotone, and
  centred at ~37 against a 40 gate.

**To pass the current acceptance rule reliably, the true per-row termination rate
must fall to ≈ 0.010.** Line must improve roughly 4×.

---

## 4. The failures pre-date ROM-B

Same policy (`model_607`), same held-out seeds 100017–100021, differing only in
ROM-B randomization scale:

| condition | line | arc | sine | circle | square |
|---|---:|---:|---:|---:|---:|
| `scale=0.0` (nominal ROM-A) | 0.0312 (clearance) | 0.0 | 0.0078 | 0.0 | 0.0078 (head) |
| `scale=1.0` (full ROM-B) | 0.0625 (clearance) | 0.0 | 0.0 | 0.0 | 0.0078 (head) |

Artifacts: `2026-07-24_11-02-18` and `2026-07-24_11-24-44`.

At **zero** ROM-B randomization the line clearance failure is already present at
60 % of the gate, with `max_clearance_error_m = 0.19566` and
`min_head_alignment = 0.98027` on square. ROM-B randomization roughly doubles the
line rate but does not create the failure mode. The remediation target is the
baseline operating point, not the uncertainty set.

### 4.1 A validation artifact worth recording

`2026-07-24_11-02-18`, `11-06-42` and `11-10-50` — the three runs cited in the run
contract as `"passed_three_consecutive_full_audits"` — produce **byte-identical**
metrics on all five shapes. The probe is fully deterministic at a fixed seed, so
these are one audit executed three times, not three independent audits. They carry
no additional statistical confidence. (`10-46-26`, two minutes earlier, differs
substantially — line clearance 0.056 vs 0.196 — under a configuration that is no
longer recoverable, since neither `PPO2_ROM_PhysX/` nor `PPO2_ROM_B_PhysX/` is
tracked in git.)

---

## 5. Three gates cannot fail

| gate | threshold | environment threshold | consequence |
|---|---:|---|---|
| `max_clearance_error_m` | `0.200` | `cutter_clearance_hard_band_m = 0.20` (`policy_4cyl/env_cfg.py:59`) | replica terminates the instant the gate would fail |
| `min_head_alignment` | `0.980` | `head_down_termination_alignment = 0.98` (`policy_4cyl/env_cfg.py:58`) | same |
| `max_post_constraint_truck_drift_m` | `0.001` | `_constrain_truck_xy` zeroes any drift above `0.001` (`policy/env.py:435-447`) | quantity is `<= 0.001` by construction |

Observed consequence in the audit: `max_post_constraint_truck_drift_m` is
**exactly `0.001000` in 159 of 160 rows**; `max_clearance_error_m` tops out at
`0.19999`; `min_head_alignment` bottoms out at `0.98002`. These are boundary
artifacts, not measurements.

So the seven-gate promotion test is effectively a four-gate test — mean error, p95
error, p95 acceleration, terminations — and terminations absorb everything the
other three would have caught. This does not make the acceptance criterion wrong
(a terminated replica is a real failure), but it does mean the reported
`FAIL: termination_rate` is uninformative about *which* physical limit was hit.
The `termination_cause_rates` field is the only place that information survives,
and section 3 uses it.

A regression test now pins this coupling:
`test_clearance_and_head_gates_coincide_with_termination_thresholds`.

---

## 6. Reward-scale audit

At the operating point, with tracking error `e ≈ 0.028 m`:

| term | formula | value at the operating point |
|---|---|---:|
| tracking | `4·exp(-(e/0.02)²)` | `+0.56` |
| progress | `0.5·clamp(Δe/0.005, ±1)` | `≈ 0` in steady state |
| action magnitude | `-0.02·mean(a²)` | `≈ -0.0002` |
| action rate | `-0.05·mean(Δa²)` | `≈ 0` |
| parent ground penalty | `-2·relu((\|c\|-0.180)/0.02)²` | `0` below `0.180`, `-0.5` at `0.190`, `-2.0` at `0.200` |
| parent head penalty | `-2·relu((0.985-align)/0.02)²` | `-0.125` at `align = 0.980` |
| ROM-B clearance margin | `-4·clamp((\|c\|-0.180)/0.020, 0, 1)²` | `0` below `0.180`, `-1.0` at `0.190`, `-4.0` at `0.200` |
| ROM-B head margin | `-4·clamp((0.985-align)/0.005, 0, 1)²` | `-4.0` at `align = 0.980` |
| ROM-B acceleration margin | `-1·clamp((a-35)/5, 0, 1)²` | `0` below `35`, `-1.0` at `40` |
| ROM-B hard event | `-20` | fires only past the gate |
| ROM-B acceleration event | `-5` | fires only past `40` |

Findings:

* **Event penalties are not too weak in magnitude.** `-20` against a maximum
  tracking reward of `+4.0` is a 5:1 ratio per step, and 36:1 at the actual
  operating point. Raising them further will not help.
* **The penalties are too late.** Clearance shaping is exactly flat for
  `|c| < 0.180`, i.e. across 90 % of the admissible range. The policy receives no
  gradient at all to move clearance from `0.17` toward `0.05`; it is only pushed
  once it is already at 90 % of the limit, into a band `0.020 m` wide. Head shaping
  is worse: the entire ROM-B head margin lives in a `0.005` window
  (`0.985 → 0.980`), and the parent head term contributes only `-0.125` at the
  limit. The result is a pile-up against the limit, which is precisely what the
  audit measures.
* **Acceleration shaping starts at 35 m/s² while the population p95 is 30–38.** The
  shaping onset is inside the operating distribution, so the term is active most of
  the time on `arc` and `circle` but saturates only in the tail.
* **The residual is barely used.** `mean_abs_residual_m ≈ 0.00018 m` against an
  authority of `0.002 m` — **9 %**. `max_cylinder4_abs_residual_m ≈ 1e-6 m` against
  `0.00025 m` — cylinder 4 is inert. The policy is not authority-limited; it is
  gradient-limited. (The residual does have real clearance authority: the
  force-zero baseline `2026-07-24_10-48-19` shows line clearance `0.127` with the
  residual disabled versus `0.056` with it enabled two minutes earlier — roughly
  70 mm of clearance effect from 2 mm of cylinder command.)

**Conclusion: strengthen the clearance and head shaping by moving its onset well
inside the operating range, not by raising the event weights.**

---

## 7. What the existing artifacts cannot answer

The audit summarises each rollout to scalars. It cannot say *where* on the path a
replica failed, what its factor vector was, or what the residual was doing. Items
3 and 4 of the remediation brief therefore need new instrumentation, which is now
in place (section 8) but has not yet been run.

Open questions, and the specific hypothesis each should test:

* **Line clearance.** `line` has excellent head alignment (`0.9958–0.9978`) and bad
  clearance (`0.185–0.19999`) at *every* `h` and *every* seed group. Because head
  alignment is good, the cutter's vertical error is unlikely to be tilt-induced;
  the hypothesis is a genuine vertical offset that grows toward one end of the
  stroke or across the direction reversal. The `path_profile` bins and the
  `path_s_m` / `reference_phase` fields on each failure record settle this directly.
  *Do not* reason about this from open-loop kinematics — that has been wrong before
  in this codebase.
* **Arc/square head alignment.** Mirror image: clearance `0.10–0.19`, alignment
  pinned at `0.980–0.981`. Hypothesis: cylinder-4 (wrist) is inert (`1e-6 m` of
  authority used), so head attitude is entirely open-loop and drifts on the shapes
  with the most heading change.
* **Circle acceleration.** Whether the `>40 m/s²` samples cluster at trajectory
  startup, at phase wrap, or at command discontinuities. `reference_phase` is
  recorded per scored sample; the per-bin `p95_acceleration_mps2` answers it.
* **truck_raw_reaction.** Present as a background second cause on every shape at
  0.008–0.03. It contributes 34/130 of line failures and 26/26 of circle failures.
  Worth confirming it is a genuine reaction force and not a reset transient.

---

## 8. Code changes made

All changes are ROM-B scoped and observational. No gate, threshold, factor range,
reward weight, or checkpoint was modified.

**`PPO2_ROM_B_PhysX/runtime/evaluation.py`**

* `_PreTerminationRecorder`: a rolling per-environment trace (default 12 steps)
  over clearance (signed), head alignment, tool acceleration, tracking error, raw
  and post-constraint truck drift, path position, reference phase, scored elapsed
  time, raw/filtered/physical residual, cylinder positions, velocities, baseline
  and corrected goals, valve command, and ROM increment. On an environment's first
  termination it snapshots the window, the classified cause(s), the simulator and
  scored step, and the environment's own six-factor vector.
* `_summarize_path_profile`: bins every scored sample by normalized path position
  and reports mean/p95/max acceleration and clearance, plus mean/min head
  alignment, per bin.
* `run_deterministic_probe` gains `collect_diagnostics=False` and
  `diagnostic_window=12`. **The flag cannot change any gate metric**: the tensors
  passed to `summarize_deterministic_probe` are the same five as before, and the
  new sample channels are appended, never substituted. With the flag off the code
  path is unchanged.

**`PPO2_ROM_B_PhysX/scripts/diagnose.py`** (new) — replays the audit's exact seed
families (`monte_carlo` offset 300000, `h_sweep` offset 400000) for a chosen subset
of shapes, rounds and `h` values with diagnostics on, and writes
`rom_b_diagnostics.json` and `rom_b_failed_environments.csv`. Shape indices are
taken from the canonical `line,arc,sine,circle,square` order so a subset run
reproduces the same seeds as the full audit. It loads a checkpoint read-only and
never writes one.

**`PPO2_ROM_B_PhysX/tests/test_rom_b_contracts.py`** — four new offline tests:
recorder window/cause/terminal-snapshot behaviour, path-profile binning, the
empty-profile edge case, and the gate/termination-threshold coupling from
section 5. All 19 tests pass.

---

## 9. Why nominal behaviour and the ROM-B physical contract are unchanged

* `rom_b_runtime.py`, `rewards.py`, `env.py`, `env_cfg.py`, `curriculum.py`,
  `train.py` and `evaluate.py` are untouched by this work.
* Factor ranges, the six-factor order, `health_scale_to_factors`, the independent
  per-environment uniform sampling, and the nominal bit-exactness path
  (`_active_factors()` returning `None` in `mode="nominal"`) are unmodified.
  `test_nominal_runtime_is_bit_exact_over_multiple_rom_ticks` and
  `test_zero_structured_factors_match_direct_model_inference` still pass.
* Factors remain private runtime state: `predict_delta` and `step` still raise on
  any attempt to pass them in, and the 42-dimensional observation is unchanged.
  The diagnostics read `true_factors` from the **runtime object during evaluation
  only**; nothing is added to the observation or the critic input.
* `model_53` is neither overwritten nor re-selected. `diagnose.py` opens it
  read-only with `optimizer/iteration/rnd` loading disabled.
