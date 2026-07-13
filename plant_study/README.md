# Plant-study diagnostics

These scripts bypass or isolate parts of the learned controller to measure the physical
Isaac/PhysX plant. They are offline diagnostics, not production control.

- `direct_drive.py`, `characterize_plant.py`: direct cylinder target response.
- `map_ground_band.py`, `refine_ground_band.py`, `verify_ground_arc.py`: settled ground
  workspace mapping.
- `measure_ground_ee_jacobian.py`: one-shot physics-measured cutter Jacobian.
- `steering_response_probe.py`: fixed steering response characterization.
- remaining `*_probe.py` files: frame, reach, limit, and cutter-point investigations.

`trajectory_direct.py` is retained only as a rejected diagnostic: its live FK
perturbations destabilize the running articulation. Do not use its results as an
acceptance gate and do not repeat that technique.
