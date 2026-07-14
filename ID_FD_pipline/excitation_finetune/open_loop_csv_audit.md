# Open-loop excitation dataset audit — consolidated

**Status: PASSED (audit complete) / BLOCKED (valve identification needs a regenerated export)**

Scope: `Excitation_Finetune.csv` (41 cols, 43,200 rows, 24 runs × 1,800, dt = 0.01 s) plus the
MAT-side ablations on `FD_cylinder_data.mat` (1,040 runs × 440, dt = 0.05 s) that answer the
two-dataset question. Companion machine-readable outputs:
`open_loop_csv_schema.json`, `open_loop_run_summary.csv`, `open_loop_sample_masks.csv`,
`open_loop_signal_stats.csv`, `open_loop_horizon_identifiability.json`,
`../models/fd/mat_fd_ablation_results.json`.

## 1. Structure (Section A) — clean

- 24 runs × 1,800 samples, dt exactly 0.01 s, time strictly monotonic, no NaN/Inf,
  no duplicate timestamps, `valid` = 1 everywhere.
- Run types decided from the actual `u` channels (not names): per parameter group of six —
  four SISO runs (u1..u4), one CH5 run (u5 only), one MIMO run. Groups: 1–6, 7–12, 13–18, 19–24.
- `exc_ch` agrees with the actually-excited channel in every SISO run.
- Last-4-column identity verified: `ddP_i(k) = dP_i(k+1) − dP_i(k)` within runs to < 1e-7.
  The CSV's last four columns are **TargetDeltadP**, as stated.

## 2. Schema verdict — the CSV is an incomplete export

Expected combined schema is **49** columns; the CSV has **41**. Missing: all four
`TargetDeltaExtension` and all four `TargetDeltaExtensionVelocity` columns. The original
excitation MAT is authoritative; the CSV is kept unchanged for comparison only.
`extract_excitation_mat.py` is the strict 49-column extractor (explicit field names,
per-run indexing, within-run k+1−k target validation, CSV round-trip, fatal on any mismatch).
**The authoritative excitation MAT is not on this machine** — extraction is blocked until it
is supplied.

## 3. Export defect — the logged u did not produce the logged state

Per-run MD5 hashes of the state block (x, xd, dP, F, q, qd):

- 24 / 24 distinct **valve** trajectories, but only **12 / 24 distinct state** trajectories.
- Within every parameter group the four SISO runs — which excite **different** valves —
  share one bit-identical state block (max |x − x_base| = 0.0 exactly).
- The CH5 runs (u1..u4 = 0) still show cylinder motion.

So the state columns were written from one simulation per group while `u` was written
per-run. The extractor reproduces this check and dies with "EXPORT DEFECT REPRODUCED"
if a regenerated export still shows ≠ 24 distinct state trajectories.

## 4. Horizon identifiability (Section C) — valve carries zero information

Method: ridge-ARX, state-history vs state+valve-history, horizons h = 1–100 samples
(0.01–1.00 s), run-wise splits only, alpha selected on a temporal validation block.
**Positive controls pass** (they must, or the detector is broken — the first, unregularized
version failed its own controls and was discarded):

- `ddP ← xd` (physically causal): ΔR² +0.003 … +0.035 ✓
- `xd ← dP` (physically causal): ΔR² +0.001 … +0.005 ✓
- `ddP ← u` (the valve): ΔR² −0.019 … +0.004 — **zero at every horizon**, including on the
  MIMO runs, which have unique state *and* unique valve.

So the decoupling is not only the duplicated SISO blocks: even where state is unique, the
logged valve does not predict it. This CSV cannot identify valve authority at any horizon.

## 5. MAT-side ablations (two-dataset question) — same null, independently

`mat_fd_ablation.py`, official run split preserved (verified: no leakage — val run state
blocks appear nowhere in train), ridge with alpha chosen on a held-out tenth of the train
runs, one-step val R² (mean over 4 cylinders):

| ablation           |     dx |    dxd |    ddP |
|--------------------|-------:|-------:|-------:|
| full (20 features) | 0.9705 | 0.3495 | 0.4051 |
| valve zeroed       | 0.9704 | 0.3466 | 0.4044 |
| valve removed      | 0.9704 | 0.3471 | 0.4045 |
| valve only         | 0.0155 | 0.0002 | 0.0007 |
| xd only            | 0.9571 | 0.0379 | 0.0850 |
| state history (lags 0–8) | 1.0000 | 0.9105 | 0.9124 |
| valve history (lags 0–8) | 0.0361 | 0.0116 | 0.0127 |
| both histories     | 1.0000 | 0.9118 | 0.9167 |

Closed-loop rollout (x/xd/dP integrated by prediction, u and F teacher-forced), extension
RMSE [m]:

| H       | full     | valve zeroed | valve removed |
|---------|---------:|-------------:|--------------:|
| 0.05 s  | 0.00303  | 0.00304      | 0.00304       |
| 0.25 s  | 0.02928  | 0.02942      | 0.02933       |
| 0.50 s  | 0.05348  | 0.05388      | 0.05354       |
| 1.00 s  | 0.07130  | 0.07230      | 0.07174       |

Deleting the valve costs ≤ 1.4 % rollout error even at a 1-second horizon. The MAT's fit is
kinematic persistence; valve authority is **not identifiable in the MAT either** — which is
exactly why the deployed FD's valve gain came out 9–19× wrong while scoring R² 0.99.

## 6. Decision and recommendation

Of the four options, the evidence selects: **the current datasets cannot fix FD's valve
sensitivity; regenerate the open-loop excitation export and identify the valve there,
keeping the MAT only for state-coverage pretraining.** Concretely:

1. **Blocked-on-user**: supply the authoritative excitation MAT (or re-log it — the harness
   must write per-run state, see §3). Run `extract_excitation_mat.py`; it must report
   24/24 distinct state hashes and pass all k+1−k target validations.
2. Re-run the Section-C ridge-ARX with positive controls on the regenerated 49-column data;
   the gate is `ddP ← u` ΔR² clearly positive at some horizon.
3. Stage A: keep MAT pretraining as-is (it teaches state persistence well — that is all it
   contains). Stage B: fine-tune FD on the regenerated open-loop data with the loss weighted
   toward future dP and xd over the trivial one-step dx, at horizons where the valve is
   identifiable. Stage C: joint fine-tune with balanced replay; the 0.01 s / 0.05 s
   timebases are handled by horizon-matched targets, never by scaling all targets by 0.01/0.05.
4. Then train ID through the frozen FD (identity-composition Jacobian + rollout loss), and
   only after all four cylinders pass hold / step / sine / coupled / rollout gates vs the
   direct-PhysX teacher, resume tool-space trajectories.

What is *not* blocked and already done on the runtime side: plant repaired (unity-gain
servo, cond 1.1), ID target-feature conventions fixed, direction guard removed (composite
cond 482 → 10.7). The single remaining structural defect is FD's valve gain, and both
existing datasets have now been shown — with controls — to be unable to constrain it.
