# ROM-B handoff — state, evidence, my errors, and what to do next

Written 2026-07-26 by Claude at the user's request, for Codex to take over.
Everything below is either (a) an artifact on disk you can re-run, or (b) marked
explicitly as inference. Section 4 lists the mistakes I made so you can check
whether any of them contaminated a conclusion I still believe.

---

## 1. Where the work stands

**ROM-B (`model_53`) is certified on termination, and fails on acceleration.**

Artifact: `reports/evaluation_2026-07-25_15-38-19/` — 250 rollouts, 128 envs,
seed 17, per-condition process isolation, frame-consistent clearance.
Contract records `condition_process_isolation: true`.

| suite | worst condition | bound | gate |
|---|---|---:|---:|
| Monte Carlo | square | 0.0437 | 0.05 PASS |
| h sweep | h=1.2 square | 0.0396 | 0.05 PASS |

All 50 conditions pass the predeclared statistic
(`max(pooled Clopper-Pearson upper, seed-group t upper)`, 95 % one-sided,
per shape / per (h, shape), worst shape controls).

**The one open defect:** 13 of 250 rows exceed the 40 m/s² acceleration gate,
max 42.88. **All 13 are the `circle` trajectory**; zero on line, arc, sine,
square. Exceedance count by h is 3, 0, 0, 2, 2, 2, 1, 1, 1 across h = 0.4→1.2,
i.e. essentially flat — this is *not* a hydraulic-degradation problem.

**The privileged oracle shows adaptation buys nothing here.**

Artifact: `reports/oracle_partial_summary.txt` (evaluation killed by the user at
190/250 rollouts; the raw rows are in `reports/eval_oracle_2026-07-25_23-32-42.log`).

| | oracle (true factors given) | ROM-B (factor-blind) |
|---|---:|---:|
| Monte Carlo worst bound | 0.0420 | 0.0437 |
| h sweep worst so far | 0.0433 (h=0.9 square) | 0.0396 (h=1.2 square) |
| acceleration over 40 | 9/190 (4.7 %) | 13/250 (5.2 %) |
| circle max acceleration | 42.41 | 42.88 |

Monte Carlo is **complete** (25/25). The h sweep covered h = 0.4–1.0 and was
killed before 1.1 and 1.2 — which is where ROM-B's own worst case sat, so the
h-sweep comparison is genuinely incomplete. Do not present it as finished.

---

## 2. The two findings that changed the diagnosis

Both were measurement defects, not policy defects. Both are fixed and tested.

### 2a. Frame-inconsistent clearance

`_compute_cutter_world_z` applies the tool offset without rotation. In the
head-down pose that puts the cutter ~1.58 m below the orientation-aware
`_compute_cutter_pos_w`. The trajectory library used the same unrotated formula,
so the two errors largely cancel — but the truck's **absolute ride height**
survives and leaks 1:1 into reported clearance (0.164 m measured, 82 % of the
0.200 m limit).

Fix: `PPO2_ROM_PhysX/policy/env.py:425` `_clearance_signal()`, gated by
`clearance_frame_consistent` (default False so historical artifacts reproduce).
Evidence: MC line seed 300017 went 7 failures → 0. The full 250-row diff between
`evaluation_2026-07-25_00-56-12` and `07-36-48` changed **exactly one column**
(`max_clearance_error_m`), every dynamics number bit-identical.

### 2b. Audit harness leaked state between rollouts

Running 250 rollouts in one process inflates termination rates. Re-running the
six failing h conditions one process each:

* `truck_raw_reaction` failures **100 → 14**
* `head_alignment` failures **50 → 47** (unchanged — this is the control)
* all six then passed, worst bound 0.0352

A 3-rollout smoke test shows the mechanism: the first rollout is bit-identical
between modes and every later one diverges further (isolated 0.02368, 0.023967,
0.025506 vs sequential 0.02368, 0.025771, 0.029034).

Fix: `--isolate_conditions` in `scripts/evaluate.py`, one subprocess per
(suite, h, shape). Contract records `condition_process_isolation`. A run without
it now prints a not-valid-for-certification warning.

**THE BIGGEST OPEN WEAKNESS — please attack this.** I measured the effect and
built a protocol around it, but **I never identified which state leaks.** I
checked that the ROM runtime, DLS/cascade, `_dP`/`_force` and truck accumulators
are all reset; the residual filter is PhysX-internal. "Simulator state leaks" is
a label, not a mechanism. If isolation is the wrong protocol, the certification
in §1 collapses. Finding the actual variable would either harden or overturn it.

Note isolation cuts **both** ways: it removed fake truck failures *and* revealed
real acceleration failures (circle over-40 rows went 3 → 13), because
contaminated rollouts terminated early and never reached the high-curvature
segment. That asymmetry is evidence the fix is doing something real, not just
making numbers smaller.

---

## 3. Why randomization taught the policy nothing

Measured, not inferred:

* Factors move the plant ~14 mm of cylinder divergence (p95 27 mm) — the
  randomization is physically real.
* Closed-loop tracking error is nonetheless **flat at 0.0307–0.0314 m across the
  entire envelope** — the DLS/cascade inner loop rejects the disturbance.
* The residual uses **8.6 %** of its 2 mm authority (max 9.5 %), and varies only
  **1.4 %** from h=0.4 to h=1.2 (0.1703–0.1727 mm). Cylinder 4 uses 0.4 %.
  Authority is *not* the binding constraint; the policy is barely a function of
  its input.
* `e_flow` was dropped 2026-07-25 as inert (median sensitivity 0.73 % of
  `e_comp`); range pinned to [0, 0], index retained for tensor compatibility.

**Structural reason, and the key point for the AAM plan:** ROM-B factors are
*fixed at reset* (`env.py`: "hidden, reset-fixed"). A plant that is constant
within an episode, underneath a converging feedback controller, is precisely the
case where online identification is worthless — the loop absorbs the offset in a
second or two. The oracle result in §1 is the direct confirmation: even *given*
the true factors, performance does not improve.

The user's goal is to show AAM outperforms ROM-B. **As configured, that
comparison is not available**, and the oracle bound proves it: AAM's estimates
are strictly worse than the truth, so if the truth buys nothing, AAM buys
nothing.

Relevant literature (I recalled these; **verify before citing** — the user's
thesis rule is real citations only): Yu et al., RSS 2017 (UP-OSI); Kumar et al.,
RSS 2021 (RMA); Peng et al., ICRA 2018 (dynamics randomization with an LSTM,
implicit system identification). The last one matters because ROM-B is a
single-frame 42-dim MLP with **no memory** — a reviewer will ask whether the
baseline was handicapped.

---

## 4. My errors this session — check whether any contaminated a conclusion

Listed so you can audit them, not as ceremony.

1. **Claimed `model_53` passes with CP95 = 0.0487** by pooling globally across h.
   Wrong — that can't certify per-h. Retracted after overdispersion analysis.
2. **Recommended n ≥ 1024 per (h, shape).** Wrong; the binding quantity is seed
   groups, not environments. The user corrected my arithmetic (2.67×, not 8×).
3. **Called USD `ComputeWorldBound` "ground truth."** It's static spawn geometry,
   never synced from PhysX headless. Retracted.
4. **Reported an "11× worse" degraded case** that came from a harness RNG bug
   (`mode="random"` consumed global RNG, so compared rollouts had different
   initial extensions). Retracted same turn.
5. **`--num_envs 4096` in a training command I gave the user.** Overflowed the
   PhysX contact buffer from t=15 s, silently corrupted dynamics (acc95 48–54 vs
   31–40; safety rate 0.13–0.15 vs 0.01–0.03), then died with NaN in
   `cylinder_q`. Every ROM-B artifact was produced at 128. Now a hard stop
   (`VALIDATED_MAX_NUM_ENVS`).
6. **Broke the evaluator with my own clearance flag.** `--clearance_frame_consistent`
   routed the env's obs/reward/dones through `_clearance_signal()` but
   `runtime/evaluation.py` kept an inline reconstruction. The audit terminated on
   one signal and reported the other. Caught because the report said
   `FAIL: ['clearance']` with zero clearance terminations — an impossible pair.
7. **Said truck reaction was "a faithful readout of the arm pulling on its
   mount."** Correlation with tool acceleration is 0.109. Withdrawn.
8. **Said 2 mm residual authority might be too little.** Utilisation is 8.6 %.
   Withdrawn.
9. **Wrote the oracle to train from scratch**, arguing a warm start would seed it
   with the factor-blind solution. Backwards — `model_53` is itself a fine-tune
   of PPO2_A, so from-scratch would differ in initialization *and* budget.
   Fixed to zero-padded PPO2_A transfer (42→47), same seed, same LR, same
   curriculum. **The oracle result in §1 used the corrected version.**
10. **Hardcoded `actor_factor_blind: True`** in the run contract. The first
    oracle run therefore recorded itself as factor-blind while feeding the true
    factors to both networks. The user caught the related stale `e_flow` banner;
    chasing it found this. That run was discarded and retrained.
11. **General pattern the user called out:** too much speculation presented with
    the same confidence as measurement. Where I've kept a claim below, it's
    because there's an artifact behind it — please verify rather than trust.

---

## 5. What I'd do next, in priority order

**1. Find the leakage mechanism (§2b).** Highest value. The certification depends
on it and I could not close it. Suggested approach: run the same seed twice in
one process and diff full env state between rollout 1 and rollout 2 — articulation
root state, joint velocities, PhysX solver residuals, the residual filter, ROM
internal state. The first rollout is provably clean (bit-identical across modes),
so the divergence appears at rollout 2 and should be localisable.

**2. Diagnose circle acceleration.** 15 minutes, no code changes needed; the path
profile already bins acceleration into 10 segments along the trajectory.

```bash
python PPO2_ROM_B_PhysX/scripts/diagnose.py --headless --device cuda:0 \
  --num_envs 128 --seed 17 --suite h_sweep --h_values 0.4 \
  --trajectory_shapes circle --rounds 1,2,3,4,5 --eval_steps 120 \
  --diagnostic_window 12 --clearance_frame_consistent \
  --checkpoint logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14/model_53_ppo2_rom_b_final.pt
```

Concentrated in one or two bins → trajectory seam or curvature discontinuity,
cheap fix. Spread across bins → genuine speed-versus-authority limit at
0.18 m/s, and the honest options are a lower path speed for circles or a
shape-specific bound. **I have not run this; both readings are hypotheses.**

**3. Decide the AAM story.** Three options, my recommendation is (a) framed by (c):

  a. **Make the plant non-stationary within an episode** — step or drift the
     health coordinate mid-rollout (seal degrading over a work cycle, oil
     warming, load change). This is the only option that creates a real case for
     online adaptation, and it matches what the RMA line of work actually
     demonstrates. Should be a new variant (ROM-B′), not an edit, so the
     certified `model_53` and its contract stay intact.
  b. **Widen h to [0.4, 1.6]** (the user floated this). Cheaper, but I expect it
     to fail the same way: the inner loop rejects a larger constant as easily as
     a smaller one. It attacks magnitude when the problem is stationarity.
  c. **Report the negative result** — "domain randomization suffices; online
     adaptation gives no measurable benefit when the plant is constant within an
     episode and a converging inner loop is present," backed by the oracle bound.
     This is a genuine finding and the oracle evidence is stronger than most
     papers offer for the positive claim.

**4. If the AAM comparison proceeds, fix the baseline.** ROM-B is memoryless.
Either give the baseline memory (a recurrent ROM-B, or the plant-response
observer already built at `runtime/plant_observer.py` — 8 EWMA features,
zero-padded transfer, **never trained or ablated**), or state the memoryless
scope explicitly up front. Otherwise "adaptation beats randomization" reads as an
architecture artifact (Peng et al.).

---

## 6. Standing constraints from the user

Do not: relax the 5 % termination gate; raise the 0.20 m clearance limit; raise
the 40 m/s² acceleration limit; change seeds until a run passes; drop failed or
incomplete replicas from statistics; replace worst-shape acceptance with pooled
averages; train only on `h`; add `h` to policy observations; multiply ROM
outputs / pressure / velocity / extension / PPO actions by `h`; change the
supported factor ranges; add unsupported `e_pressure` / valve-delay / clearance /
pressure-limit randomization; expose true ROM-B factors to actor or critic
(**the privileged oracle deliberately violates this — it is a diagnostic, is
recorded `certifiable: false`, and lives in its own log tree
`ppo2_rom_b_privileged_oracle/`; it must never be presented as a ROM-B result**);
increase residual authority without a controlled audit; overwrite `model_53`;
start remediation from `model_607`; modify ROM-C or AAM; regenerate AAM data.

Training must continue to sample the factors independently.

Note also: ROM-C does **not** consume a ROM-B checkpoint. It starts from PPO2_A
`model_607` and widens 42 → 49 (`PPO2_ROM_C_PhysX/checkpoint_transfer.py:1`).
ROM-B and ROM-C are parallel branches off PPO2_A, not sequential stages.

---

## 7. Test suite

`PPO2_ROM_B_PhysX/tests/test_rom_b_contracts.py` — **53 tests, all passing**.
Each test that encodes a hard-won fact states the measurement in its docstring.
Run with the `env_isaaclab` conda environment; tests that would import Isaac Lab
parse source instead.
