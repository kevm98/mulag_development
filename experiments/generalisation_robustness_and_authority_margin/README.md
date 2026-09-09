# Generalisation, mechanical robustness and authority margin

Thesis §7.4. Actuator authority is reduced by scaling realised actuator
displacement, `Δx_applied = κ · Δx_actuator`, `κ ∈ (0,1]`, applied *after* each
stack's actuator representation so the comparison does not depend on the
individual factor parameterisations.

**Produces:** `figures/ch7_kappa_stacks.pdf` — lateral tracking and ground
contact as κ falls, on mesh 1337 at 4.0 km/h, at nominal and 4.0× payload.

    code/kappa_sweep.sh          run the sweep at nominal payload
    code/kappa_loaded.sh         run it under load
    code/kappa_report.py         the onset-κ table
    code/make_kappa_figure.py    builds ch7_kappa_stacks from outputs/kappa_loaded

Run from the repository root. Reads `outputs/kappa` and `outputs/kappa_loaded`.

The scope limit stated in §7.4 applies: on the inverse and forward path the
hydraulic factors are not applied at all, so the variation reaching PPO-ID-FD is
mechanical only. Nothing here supports a claim of robustness to hydraulic
variation.
