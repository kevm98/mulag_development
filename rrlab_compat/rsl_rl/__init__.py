"""RSL-RL config classes, from the installed IsaacLab when it is new enough.

The env configs here use the rsl-rl 5.x model API (``RslRlMLPModelCfg``, and
``OnPolicyRunner.load(..., load_cfg=...)``), and every checkpoint under ``logs/``
stores separate ``actor_state_dict``/``critic_state_dict`` tensors to match.
IsaacLab only ships that API from 3.0 onwards, but IsaacLab 3.0's *core*
requires torch 2.14 / numpy 2.x, which Isaac Sim 5.1 (torch 2.7, numpy 1.26)
cannot host. The RL config layer has no such constraint -- it needs only
``isaaclab.utils.configclass`` and torch -- so a copy is vendored here and used
whenever the installed IsaacLab is too old.

Import from this module rather than ``isaaclab_rl.rsl_rl`` directly, so the repo
keeps working on both an IsaacLab 2.3.x and an IsaacLab 3.x machine without any
edits, and without depending on local modifications to the IsaacLab checkout.
"""

from __future__ import annotations

# Checked here rather than at the call site: without it, a mismatched rsl-rl
# surfaces as `TypeError: load() got an unexpected keyword argument 'load_cfg'`
# thirty seconds into an Isaac Sim launch, which reads like a repo bug.
def _require_rsl_rl_5() -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("rsl-rl-lib")
    except PackageNotFoundError:
        raise ImportError(
            "rsl-rl-lib is not installed. This repo needs rsl-rl-lib==5.0.1:\n"
            "    python -m pip install 'rsl-rl-lib==5.0.1'"
        ) from None
    if int(installed.split(".")[0]) < 5:
        raise ImportError(
            f"rsl-rl-lib {installed} is too old for this repo, which needs the 5.x "
            "model API (separate actor/critic, OnPolicyRunner.load(load_cfg=...)). "
            "Every checkpoint under logs/ is stored in that layout, so 3.x cannot "
            "load them. Fix with:\n"
            "    python -m pip install 'rsl-rl-lib==5.0.1'"
        )


_require_rsl_rl_5()

try:
    # IsaacLab >= 3.0: use what is installed, so we follow upstream fixes.
    from isaaclab_rl.rsl_rl import (  # noqa: F401
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )

    USING_VENDORED = False
except ImportError:
    # Taking the whole set from one source matters: the 2.3.x wrapper and the
    # 3.x runner config disagree about how observations reach the policy.
    from ._vendor.rl_cfg import (  # noqa: F401
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )
    from ._vendor.vecenv_wrapper import RslRlVecEnvWrapper  # noqa: F401

    USING_VENDORED = True

__all__ = [
    "RslRlMLPModelCfg",
    "RslRlOnPolicyRunnerCfg",
    "RslRlPpoAlgorithmCfg",
    "RslRlVecEnvWrapper",
    "USING_VENDORED",
]
