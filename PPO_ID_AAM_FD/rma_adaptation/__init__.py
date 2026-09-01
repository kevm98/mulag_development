"""CPU-safe building blocks for the RMA/A-RMA PPO preparation.

This package deliberately has no IsaacLab or simulator import.  It describes
the contract that a future GPU rollout must wire into the existing PPO
environment, and makes the parts that can be audited offline executable on
CPU: context normalization, causal history, actuation stress dynamics,
checkpoint expansion, runtime modes, and analysis metrics.
"""

from .core import (
    CONTEXT_DIM,
    CONTEXT_NAMES,
    ActuationConfig,
    ActuationStressWrapper,
    DeploymentContext,
    DeploymentActuation,
    NativeRigidBodyWiring,
    causal_mismatch_features,
    normalized_context,
    verify_native_mass_com_wiring,
)
from .modes import ContextMode, assemble_policy_observation, assert_no_context_leakage
# NOTE: this package historically exported the CPU smoke skeleton under the
# plain names, while PPO_AAM_FD exports the real phi from `core`. The
# explicit names below make that divergence visible at the import site.
from .online import (  # noqa: F401
    SmokeAAMEstimator as AAMEstimator,
    SmokeAAMTrainer as AAMTrainer,
    OnPolicyHistoryRecorder,
)
from .metrics import (
    delta_adapt,
    opportunity_grid,
    plot_recovery_curve,
    recovery_fraction,
    recovery_plot_data,
    stress_thresholds,
)
from .config import DEFAULT_RMA_CONFIG, Phase1Config, Phase2Config, Phase3Config, RMAExperimentConfig

__all__ = [
    "CONTEXT_DIM", "CONTEXT_NAMES", "ActuationConfig", "DeploymentContext",
    "DeploymentActuation", "ActuationStressWrapper", "NativeRigidBodyWiring", "causal_mismatch_features",
    "normalized_context", "verify_native_mass_com_wiring", "ContextMode",
    "assemble_policy_observation", "assert_no_context_leakage", "AAMEstimator",
    "AAMTrainer", "OnPolicyHistoryRecorder", "delta_adapt", "opportunity_grid",
    "plot_recovery_curve", "recovery_fraction", "recovery_plot_data",
    "stress_thresholds", "DEFAULT_RMA_CONFIG", "Phase1Config", "Phase2Config",
    "Phase3Config", "RMAExperimentConfig",
]
