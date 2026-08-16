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
| **PPO--ID(AAM)--FD** | policy → desired motion → frozen ID + adapter(ẑ) → frozen strict FD → PhysX | causal AAM history → ẑ |

Use the `\ppoROM`, `\ppoIDFD` and `\ppoIDAAMFD` macros from `macros.tex`.
The labels PPO1, PPO2, PPO1-A/B and PPO2-A/B/C/D are **superseded** and must
not reappear. Two further rules follow from the plan: the AAM does **not**
condition the ROM (it conditions a bounded adapter around the frozen ID;
conditioning the policy directly on ẑ is an ablation), and the policy is
**not** a residual policy — it emits the desired cylinder command directly,
with no nominal command added at runtime.

## Structure

- `main.tex` — main file: front matter (title page, second page, declaration,
  acknowledgements, abstract, table of contents), chapter includes, appendices,
  bibliography.
- `packages.tex` / `macros.tex` — extra packages and own macros (nomenclature
  shorthands, stack-name macros, `\todo{}` marker).
- `literature.bib` — BibTeX bibliography.
- `chapters/`
  1. `ch1_introduction.tex` — motivation, verge-tracking problem, limitations
     of fixed learned actuators, RQ1–RQ5, objectives, contributions stated
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
  8. `ch8_conclusion_outlook.tex` — direct answers to RQ1–RQ5, demonstrated
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

## TODOs before submission

- Fill in author full name, supervisor, issue/submission dates in `main.tex`.
- Every section body is a `\todo{...}` placeholder with the planned content as
  a comment above it (taken from the experimental plan).
