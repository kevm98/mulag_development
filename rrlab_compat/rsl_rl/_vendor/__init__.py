"""Verbatim copies of ``isaaclab_rl.rsl_rl`` from IsaacLab v3.0.0-beta2.patch1.

Upstream: https://github.com/isaac-sim/IsaacLab (BSD-3-Clause, headers intact).
Do not edit these by hand -- re-copy them from a newer IsaacLab tag instead, so
diffs against upstream stay readable.

IsaacLab 3.0's own ``__init__.py`` is deliberately not copied: it calls
``isaaclab.utils.module.lazy_export``, which does not exist in IsaacLab 2.3.x.
The parent package imports these modules directly instead.
"""
