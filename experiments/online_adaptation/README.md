# Online adaptation

Thesis §7.5. Within-stack evaluation of PPO-AAM-FD against its own frozen base,
across payload, on held-out meshes.

**Produces:** `figures/ch7_adaptation_forest.pdf` — frozen and adaptive
performance on held-out mesh 3141, per payload level.

    code/final_eval.sh           score the arms            -> outputs/final
    code/heldout8191.sh          reserved-mesh repetition  -> outputs/heldout8191
    code/aa_floor_p4.sh          A/A control               -> outputs/aa_floor
    code/final_table.py          pooled paired median table
    code/heldout8191_table.py    mesh 8191 repetition table
    code/robust_table.py         robustness table
    code/wronginfo_table.py      correct against wrong payload
    code/make_adaptation_figure.py   builds the forest figure from outputs/final

Run from the repository root. AF arms are trained under
`PPO_AAM_FD/experiment_scripts`; this folder scores them and builds the reported
tables and figure.

Read `mass_scale_mean` from the trace before analysing: the plant must be
confirmed to have done what was asked. Paired comparisons only, with the sign
count reported alongside the mean.
