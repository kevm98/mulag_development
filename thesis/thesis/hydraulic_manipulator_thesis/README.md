# Learning-Based Online Adaptation for Control of a Hydraulic Manipulator

Master thesis using the RRLab LaTeX class (`template/rrlab.cls`), structured
after `docs/Hydraulic_Manipulator_Thesis_Experimental_Plan.pdf` (full rebuild
v2, 9 August 2026).

## Nomenclature (authoritative)

The final comparison has exactly three stacks, and these are the only names
the thesis may use for them:

| Stack | Actuation path | Online adaptation |
|---|---|---|
| **PPO--ROM** | policy → desired cylinder motion → command ROM → PhysX | none |
| **PPO--ID--FD** | policy → desired motion → frozen ID → frozen strict FD → PhysX | none |
| **PPO--(AAM)--ID--FD** | policy(obs+ẑ) → desired motion → frozen ID + adapter(ẑ) → frozen strict FD → PhysX | causal AAM history → ẑ |

**ẑ routing (authoritative, verified against `PPO_ID_AAM_FD/env_cfg.py` and
`mulag_plant/runtime/mulag_upper_level_env.py`):** ẑ → {π_θ, g_ψ}. It is appended
to the policy observation (66 → 68, `online_aam_policy_observation=True`) *and*
conditions the bounded ID adapter. Policy conditioning is **not** an ablation —
it is the deployed design. Consequently the stack never reduces to PPO--ID--FD:
zeroing the residual restores the non-adaptive *actuator path* only; the actor
remains the 68-input latent-conditioned network.

Use the `\ppoROM`, `\ppoIDFD` and `\ppoAAMIDFD` macros from `macros.tex`.
The labels PPO1, PPO2, PPO1-A/B and PPO2-A/B/C/D are **superseded** and must
not reappear. Two further rules follow from the plan: the AAM does **not**
condition the ROM, and the policy is
**not** a residual policy — it emits the desired cylinder command directly,
with no nominal command added at runtime.

## Writing rules (authoritative)

Verified against `../template_latex_howto/howto.tex` and the accepted example
`figure_references/thesis_sample.pdf`.

- **Abstract: no numbers, and keep it short.** Qualitative prose only. Figures
  belong in the chapters.
- **No em-dashes (`---`) in prose.** Use a comma, a full stop, or "namely".
  The only legitimate `---` is a table cell meaning "not applicable"; leave those
  alone. `--` in `PPO--ROM` and in ranges is fine.
- **No flowery language.** No rhetorical slogans, no "not X but Y", no dramatic
  colons. State the result, state the evidence, move on.
- **Section 1.4 is "Contribution of This Thesis"** and 1.5 is "Thesis Structure".
  Contributions and research questions are **bold run-in headings with the colon
  inside** (`\textbf{Name:} text...`), never `\subsection*` blocks, `description`
  lists, or `C1 --- ` labels.
- **Leave the hyperlink frames alone.** No `\hypersetup{hidelinks}`.
  `rrlab.cls` applies `hidelinks` only for dissertations (`\ifdissfinal`); the
  red/green/blue link boxes are correct for a Master thesis and appear in
  `thesis_sample.pdf`. If the look ever needs to change, use the class option
  `colorlinks` (`rrlab.cls:29`), not a raw `\hypersetup`.
- **Section 1.5 is one bold `Chapter N` paragraph per chapter**, after a lead
  paragraph ending "The document is organized as follows:", then one paragraph
  per appendix. Do not compress it into a single paragraph, however tempting;
  the accepted sample devotes roughly two pages to it.
- **Voice** follows `thesis_sample.pdf`: measured academic register, frequently
  passive ("is shown to", "are established as"), no reader instructions.

## Framing (authoritative)

This is a **learning-based hydraulic control and model-transfer thesis**, not an
AAM thesis. Online adaptation is **RQ6**, the last of six questions, and it is
asked only after RQ1-RQ5 have established that the actuator models transfer,
that a policy can control through them, and how much plant variation the loop
already absorbs. A small closed-loop adaptation result is therefore a finding
about the system, not a failure of the work: good models plus a robust policy
leave less for adaptation to recover.

Do not restructure the argument so that the thesis stands or falls on whether
the AAM improved PPO. The ID/FD transfer result, the command-authority failure
and its excitation fix, the ROM, and the policy-only control stack are
contributions in their own right (C1-C4); adaptation belongs to C5.

## Structure

- `main.tex` — main file: front matter (title page, second page, declaration,
  acknowledgements, abstract, table of contents), chapter includes, appendices,
  bibliography.
- `packages.tex` / `macros.tex` — extra packages and own macros (nomenclature
  shorthands, stack-name macros, `\todo{}` marker).
- `literature.bib` — BibTeX bibliography.
- `chapters/`
  1. `ch1_introduction.tex` — motivation, verge-tracking problem, limitations
     of fixed learned actuators, RQ1–RQ6, objectives, five contributions stated
     according to completed evidence, scope (6–9 pages)
  2. `ch2_fundamentals_related_work.tex` — hydraulic actuation, DLS, learned
     inverse/forward dynamics, reduced-order modeling, domain randomization
     versus adaptation, system identification, RMA-style context inference,
     structured priors (residual RL and distillation), PPO, safe evaluation;
     ends with the research gap (12–18 pages)
  3. `ch3_system_fundamentals_problem_formulation.tex` — FME 500 / PhysX
     plant, four cylinder coordinates, the moving-verge primary task and the
     parked diagnostic task, multi-rate timing, observable versus privileged
     signals, constraints, formal tracking and robustness objectives, the
     eight-gate contract (10–15)
  4. `ch4_data_generation_preparation.tex` — historical Simscape provenance
     for ROM/ID/FD, the new IsaacLab 118-run collection, DLS and PRBS
     excitation, factor ranges, safety gates, 50 ms alignment, run-level
     splitting, design adequacy (10–16)
  5. `ch5_learned_models_online_adaptation.tex` — command ROM and ID/strict-FD
     contracts; SD₀ (§5.5); factor encoder and SD_z; the latent-matters
     result; AAM dataset, history selection and downstream validation; the
     latent-conditioned ID adapter. Makes clear that SD is **not** the
     deployed actuator (14–22)
  6. `ch6_control_framework_simulation_integration.tex` — the three stacks,
     bounded ID adapter, causal AAM update and fallback, 66-D observation,
     four-cylinder action mapping, bounded reward, clipped PPO objective,
     teacher distillation and the matched training protocol (12–18)
  7. `ch7_experimental_evaluation_discussion.tex` — model rollouts, ID
     transfer, system response, delay, nonlinearity, end-effector/mowing
     performance, robustness, adaptation ablations, statistics, and the
     mechanisms behind positive and negative findings (18–28)
  8. `ch8_conclusion_outlook.tex` — direct answers to RQ1–RQ6, demonstrated
     contributions, limitations, future work (4–7)
  - `appendices.tex` — A: system parameters and conventions; B: dataset
    manifests and factor ranges; C: model architectures, normalizers and
    checkpoint hashes; D: PPO/IsaacLab configuration; E: supplementary
    results.

## Claim discipline

- Do not describe the AAM, the adapter, or the adaptive-policy result as
  complete before the remaining stages pass and the closed-loop experiment is
  run.
- Privileged-latent evidence is not evidence that the AAM works, and robust
  training across randomization is not online adaptation.
- Never give physical factors, the true latent, future state, hidden ROM state
  or hidden pressure to a deployable policy or estimator input.
- Report exact timing at each layer rather than one "control frequency".
- Include the negative results: ID–FD's contact failure, the tracking cost of
  PPO fine-tuning after distillation, and the plant-capability ceiling.

## Build

Requires a TeX distribution (not installed on this machine; e.g.
`sudo apt install texlive-full`, or use Overleaf):

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Replacement figure regeneration

The Chapter 2--5 replacements requested during figure revision are generated
from the editable source script below. It writes PDF/SVG line art, 600-dpi
preview PNGs, cropped USD-render outputs, and tight vector re-frames of the
existing Chapter-4 data figures without changing their data.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_figures
export MPLCONFIGDIR=/tmp/mplconfig
python scripts/make_thesis_replacement_figures.py
python scripts/make_ch5_figures.py
pdflatex main
pdflatex main
```

The conceptual roadside scene is explicitly labelled as explanatory only in
Chapter 2; it is not used as an experimental result.

## Status

Chapters 1–8 are written in full, and Appendices A–E are written. No
`\todo{...}` placeholders remain in the chapter or appendix bodies.

## TODOs before submission

- `main.tex` line 26 — date of issue (currently `TODO: date of issue`).
- `main.tex` line 29 — date of submission (currently `TODO: date of submission`).
- `main.tex` line 35 — supervisor (currently `TODO: supervisor`). The first
  reviewer is already set to Prof. Dr. Karsten Berns.
- `main.tex` lines 59–61 — acknowledgements (currently `TODO: acknowledgements`).

The author name is already set to Kevin Kudakwashe Murera.

### Known open items in the work itself

These are stated in the thesis rather than hidden, and are listed here so they
are not mistaken for oversights:

- No stack passes all eight acceptance gates under the Chapter 7 protocol; all
  three fail the lateral gate on the worst trajectory family. Chapter 7 states
  the consequence and Chapter 8 lists repairing the contract as unfinished work.
- The `hard_impacts` gate is passed by a do-nothing policy and is labelled a
  wear constraint rather than a task gate (Chapter 3).
- The AAM observation contract is pressure-inclusive; the stricter deployable
  no-pressure variant is unquantified (Chapter 7, threats to validity).
