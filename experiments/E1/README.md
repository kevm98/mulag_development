# E1 — Closed-loop task performance

**Defends:** *Control* · **Covers:** PPO-ROM, PPO-ID-FD
**Status:** complete for the matched pair · thesis §7.2, §7.3.5. The
PPO-AAM-FD rows were scored on the retired ID-adapter architecture and are
withdrawn rather than reported (§7.1.3).

Every other result in the thesis is model-level prediction error. This is the
only measurement of control.

## Run

    bash experiments/E1/code/run_e1_tracking.sh
    python experiments/E1/code/summarise_e1.py --root outputs/e1_tracking

Figures: `experiments/E1/code/make_ch7_figures.py` (writes into the thesis
`images/`; also produces the E2 and E4 figures).

## Unit of analysis — read before quoting any number

The evaluation is **deterministic**. Terrain is fixed by the mesh id, the policy
acts deterministically, initial conditions do not vary. Five evaluation seeds
returned **bit-identical numbers to nine decimal places**. `--seed` is recorded
in the report and changes nothing; the gate code says so itself:

> acceptance needs >= 3 distinct terrain meshes... varying `--seed` alone
> returns bit-identical numbers.

The unit is therefore the **(mesh, family) cell** — 3 meshes x 11 trajectory
families = **33 paired conditions**. Metrics that exist only per mesh (impacts,
head alignment, tool acceleration) are n=3 and labelled as weak evidence.

Lateral and cut-height are reported as the **mean over families**, not the worst
family: worst-family is a maximum dominated by `double_s_curve` at ~20x a
typical family, so it measures one hard trajectory rather than tracking quality.

## Traps

- **`evaluate_gates.py` hangs** when handed several meshes in one process: it
  finishes the first, re-creates the environment for the second, then spins on
  one core indefinitely. Always go through `PPO_ROM/scripts/score_per_mesh.sh`
  (one mesh per process + stall watchdog). `PARALLEL_MESHES=3` fits in 6 GiB.
- **Scoring horizon.** `select_checkpoint.sh` defaults to `STEPS=480` = 8.9 m of
  the 26 m path. Every closed-loop result must name its checkpoint AND its
  horizon. See `results/fullpath_reselection.json`.

## Result

No contrast between PPO-(AAM)-ID-FD and PPO-ID-FD reaches significance over 33
cells; every paired bootstrap interval straddles zero. H3 is neither supported
nor rejected. Mean and median disagree on cut-height and contact — the adaptive
stack is better in ~25/33 cells yet worse on the mean, a tail effect.

The **lateral gate is failed by all three stacks** on worst-mesh statistics,
PPO-ROM most severely, so it reflects the hardest trajectory family rather than
a defect of any one architecture.
