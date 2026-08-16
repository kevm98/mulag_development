# Stage 25 — Select AAM history length

Status: **causal re-score complete** (2026-08-09).

Trained a small AAM (`z_hat = AAM(H_t)`) per candidate lag (5/10/20/40 ticks
= 0.25/0.5/1.0/2.0 s) on training runs, then scored each by downstream
`SD_z(s, u, z_hat)` H20 rollout RMSE on the validation runs. The rollout uses
the AAM prediction at the causal window ending at each rollout start; it does
not average future held-out windows into an early estimate. Test split unused.
Selection used paired bootstrap 95% CIs vs the best lag and chose the shortest
acceptable candidate.

Corrected result: **L5 (0.25 s)** is the shortest acceptable lag; L10 has the
lowest validation mean, but all four candidates are within uncertainty of the
best. The earlier result selecting **L40 (2.0 s)** was invalid because it used
a noncausal full-run mean of held-out predictions. The first
run of this sweep (see `outputs/aam_stage25_buggy_zhat/`) fed RAW windows
into the per-run `z_hat` inference while the model was trained on normalized
windows, which degraded long-history latents and incorrectly selected L5.
After fixing `tools/train_aam.py` to normalize windows for inference and
correcting the evaluation to use causal per-tick `z_hat`, all four candidates
are acceptable under the predeclared rule; the shortest is selected.

Corrected artifacts: `../outputs/aam_stage25_causal/` (`rollout_L*.json`,
`selection.json`, and causal per-tick predictions). The older
`../outputs/aam_stage25/` rollout reports are historical noncausal evidence.
Tools: `tools/train_aam.py`, `tools/predict_aam.py`,
`tools/select_aam_history.py`, `tools/run_stage25_history_sweep.sh`.

Follow-up: a predeclared 20-prediction trailing mean is strictly past-only and
improves the validation score slightly. It still selects L5 under the same
shortest-acceptable rule. Reproduction: `tools/run_stage25_smoothing_eval.sh`;
artifacts: `../outputs/aam_stage25_rolling20/`.
