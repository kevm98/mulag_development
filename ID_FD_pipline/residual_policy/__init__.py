"""Residual PPO policy for the MULAG hydraulic mower arm.

The policy adds small residual corrections (for cylinders 1-3 only) to the
existing deterministic head-down A->B->A arm controller. Cylinder 4 remains
at the head-down teacher value, and the truck remains completely stationary.
"""

__all__: list[str] = []
