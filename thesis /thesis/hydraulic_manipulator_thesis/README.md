# Learning-Based Online Adaptation for Control of a Hydraulic Manipulator

Master thesis skeleton using the RRLab LaTeX class (`template/rrlab.cls`),
structured after `Hydraulic_Manipulator_Thesis_Experimental_Plan(2).pdf` (Part II).

## Structure

- `main.tex` — main file: front matter (title page, second page, declaration,
  acknowledgements, abstract, table of contents), chapter includes, appendices,
  bibliography.
- `packages.tex` / `macros.tex` — extra packages and own macros (nomenclature
  shorthands, `\todo{}` marker).
- `literature.bib` — BibTeX bibliography.
- `chapters/`
  1. `ch1_introduction.tex` — motivation, problem statement, RQ1–RQ4,
     objectives, contributions, scope, thesis structure (6–9 pages)
  2. `ch2_fundamentals_related_work.tex` — theory + literature, ends with the
     research gap (12–18 pages)
  3. `ch3_system_fundamentals_problem_formulation.tex` — platform, kinematics,
     Simscape, IsaacLab, hierarchical architecture, problem formulation (10–15)
  4. `ch4_data_generation_preparation.tex` — randomized parameter space,
     closed-loop + open-loop datasets, features, splits, diagnostics (10–16)
  5. `ch5_learned_models_online_adaptation.tex` — ID, FD, ROM, latent
     representation, AAM, training protocol, offline evaluation (14–22)
  6. `ch6_control_framework_simulation_integration.tex` — DLS, residual PPO,
     PPO1-A/B, PPO2-A/B/C, oracle diagnostic, integration, constraints (12–18)
  7. `ch7_experimental_evaluation_discussion.tex` — hypotheses, experiment
     matrix, model-level and closed-loop experiments, ablations, discussion,
     threats to validity (18–28)
  8. `ch8_conclusion_outlook.tex` — summary, answers to RQs, contributions,
     limitations, future work (4–7)
  - `appendices.tex` — A: system parameters; B: dataset/randomization;
    C: model architectures; D: PPO/IsaacLab config; E: supplementary results.

## Build

Requires a TeX distribution (not installed on this machine; e.g.
`sudo apt install texlive-full`, or use Overleaf):

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## TODOs before submission

- Fill in author full name, supervisor, issue/submission dates in `main.tex`.
- Every section body is a `\todo{...}` placeholder with the planned content as
  a comment above it (taken from the experimental plan).
