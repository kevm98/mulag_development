# Thesis Tracking Table (Notion-ready)

> **How to use in Notion**
>
> **Option A — Quick table (fastest):** Copy the whole Markdown table below (everything between the `---` markers), then paste it into a blank Notion page. Notion auto-converts it into a Table block. You can then add/remove rows and edit cells directly.
>
> **Option B — Proper database (recommended for long theses):** In Notion create a new **Database** (inline or full page) and add these **properties**:
> - **Chapter** (Number or Select)
> - **Title** (Title)
> - **Section** (Rich text)
> - **Status** (Select: Not started / In progress / Draft / Review / Final)
> - **Progress** (Number, percent format)
> - **Target words** (Number)
> - **Current words** (Number)
> - **Deadline** (Date)
> - **Priority** (Select: Low / Medium / High)
> - **Key files** (Rich text / URL)
> - **Notes** (Rich text)
>
> Then paste the row contents below into it. A database lets you filter by Status, sort by Deadline, and make a calendar view — much better for tracking "everything".

---

| # | Chapter / Section | Status | Progress % | Target words | Current words | Deadline | Key files / Resources | Notes |
|---|-------------------|--------|-----------|--------------|---------------|----------|----------------------|-------|
| 1 | **Introduction** | Not started | 0 | 1500 | 0 |  |  |  |
| 1.1 | Motivation — MULAG Saugbagger & hydraulic arm automation | Not started | 0 | 300 | 0 |  | [`PPO_ROM/README.md`](../PPO_ROM/README.md) | Why the FME 500 / vacuum excavator arm |
| 1.2 | Problem statement & scope | Not started | 0 | 300 | 0 |  |  |  |
| 1.3 | Research questions | Not started | 0 | 300 | 0 |  |  |  |
| 1.4 | Contributions | Not started | 0 | 300 | 0 |  |  |  |
| 1.5 | Thesis outline | Not started | 0 | 300 | 0 |  |  |  |
| 2 | **Fundamentals & Literature Review** | Not started | 0 | 3000 | 0 |  |  |  |
| 2.1 | Hydraulic actuation & excavator arm kinematics | Not started | 0 | 600 | 0 |  | [`verge_task/actuators.py`](../verge_task/actuators.py) | Cylinder model, kinematics |
| 2.2 | Simulation (Isaac Lab) & sim-to-real transfer | Not started | 0 | 600 | 0 |  |  | Physics sim, domain randomization |
| 2.3 | Reinforcement learning (PPO) | Not started | 0 | 600 | 0 |  |  | Policy gradient, curriculum |
| 2.4 | System identification — inverse/forward dynamics (ID/FD) & reduced-order models (ROM) | Not started | 0 | 600 | 0 |  | [`id_fd_model/README.md`](../id_fd_model/README.md) | Paired ID/FD residual pipeline |
| 2.5 | Domain adaptation / learned adaptation modules (AAM) | Not started | 0 | 600 | 0 |  | [`AAM_module/stage20_system_factor_encoder.md`](../AAM_module/stage20_system_factor_encoder.md) | Latent system-factor encoding |
| 3 | **System & Simulation Setup** | Not started | 0 | 2500 | 0 |  |  |  |
| 3.1 | MULAG plant & USD assets | Not started | 0 | 500 | 0 |  | [`extensions/rrlab_assets/rrlab_assets/mulag.py`](../extensions/rrlab_assets/rrlab_assets/mulag.py) |  |
| 3.2 | Kinematic & hydraulic modeling of the arm | Not started | 0 | 700 | 0 |  | [`PPO_ID_FD/env_cfg.py`](../PPO_ID_FD/env_cfg.py) |  |
| 3.3 | Trajectory design (2D, line/serpentine/sweep, stop-and-mow) | Not started | 0 | 700 | 0 |  | [`mulag_plant/policy/TRAJECTORY_2D.md`](../mulag_plant/policy/TRAJECTORY_2D.md), [`verge_task/trajectory_2d.py`](../verge_task/trajectory_2d.py) |  |
| 3.4 | Training environment & reward setup | Not started | 0 | 600 | 0 |  | [`PPO_ID_FD/env.py`](../PPO_ID_FD/env.py) |  |
| 4 | **Methods** | Not started | 0 | 4000 | 0 |  |  |  |
| 4.1 | Baseline controller (DLS + paired ID/FD) | Not started | 0 | 800 | 0 |  | [`PPO_ID_FD/README.md`](../PPO_ID_FD/README.md) | CPU DLS, measured cylinder feedback |
| 4.2 | PPO residual policy on ID/FD | Not started | 0 | 800 | 0 |  | [`PPO_ID_FD/scripts/train.sh`](../PPO_ID_FD/scripts/train.sh) | Curriculum, obs 33-D |
| 4.3 | ROM-based policy (PPO2) | Not started | 0 | 800 | 0 |  | [`PPO_ROM/README.md`](../PPO_ROM/README.md) | 41-D obs, exact+procedural mix |
| 4.4 | AAM domain adaptation (latent factor encoder) | Not started | 0 | 800 | 0 |  | [`PPO_ID_AAM_FD/online_runtime.py`](../PPO_ID_AAM_FD/online_runtime.py) | Stage 20–28 chain |
| 4.5 | Deployment pipeline (stage 31 ROM) | Not started | 0 | 800 | 0 |  | [`deployment_adaptation/stage31_rom_pipeline.sh`](../deployment_adaptation/stage31_rom_pipeline.sh) | Online adaptation protocol |
| 5 | **Experiments & Evaluation** | Not started | 0 | 2000 | 0 |  |  |  |
| 5.1 | Training configurations & hardware | Not started | 0 | 500 | 0 |  | [`PPO_ID_FD/scripts/preflight.py`](../PPO_ID_FD/scripts/preflight.py) | RTX 3060, num_envs 64–256 |
| 5.2 | Evaluation metrics (mean/p95 tool error, acceleration, head alignment, truck drift) | Not started | 0 | 700 | 0 |  | [`reports/PPO_PIPELINES_2026-08-11.md`](../reports/PPO_PIPELINES_2026-08-11.md) |  |
| 5.3 | Ablations & baselines | Not started | 0 | 800 | 0 |  |  | Zero-policy, model selection |
| 6 | **Results** | Not started | 0 | 2500 | 0 |  |  |  |
| 6.1 | Baseline vs. RL policies | Not started | 0 | 800 | 0 |  |  |  |
| 6.2 | Sim-to-real / deployment transfer | Not started | 0 | 900 | 0 |  | [`deployment_adaptation/reports/MINIMUM_ONLINE_ADAPTATION_PROTOCOL_2026-08-10.md`](../deployment_adaptation/reports/MINIMUM_ONLINE_ADAPTATION_PROTOCOL_2026-08-10.md) |  |
| 6.3 | AAM adaptation results | Not started | 0 | 800 | 0 |  | [`AAM_module/results/`](../AAM_module/results/) |  |
| 7 | **Discussion** | Not started | 0 | 1500 | 0 |  |  | Limitations, failure cases |
| 8 | **Conclusion & Future Work** | Not started | 0 | 800 | 0 |  |  |  |
| 9 | **Appendix** (configs, extra runs, reproducibility) | Not started | 0 | 1000 | 0 |  |  |  |
| 10 | **References / Bibliography** | Not started | 0 | 0 | 0 |  |  | Keep Zotero/Mendeley synced |
| — | **Total** |  | 0 | **17800** | 0 |  |  |  |

---

## Quick status legend

- **Not started** — haven't touched it yet
- **In progress** — actively writing
- **Draft** — full draft exists, needs editing
- **Review** — sent for feedback / self-review
- **Final** — done, formatting only

## Tips for Notion

1. Use the **Table** block (Option A) if you just want a checklist; use a **Database** (Option B) if you want calendar/filter views.
2. Link each Chapter/Status cell to an actual Notion page per chapter so "everything" lives in one place.
3. Add a **Formula** property (`Current words / Target words`) to auto-compute progress from word counts.
4. Duplicate the table and make a **Template button** per chapter so each chapter gets the same structure.
