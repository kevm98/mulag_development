"""RMA / A-RMA adaptation contract. See `core` for the design rationale."""

from PPO_AAM_FD.rma_adaptation.core import (  # noqa: F401
    CONTEXT_DIM,
    CONTEXT_NAMES,
    AAMEstimator,
    AAMTrainer,
    ActuationConfig,
    ContextMode,
    DeploymentActuation,
    DeploymentContext,
    NativeRigidBodyWiring,
    OnPolicyHistoryRecorder,
    assemble_policy_observation,
    assert_no_context_leakage,
    causal_mismatch_features,
    delta_adapt,
    normalized_context,
    recovery_fraction,
    stress_thresholds,
    verify_native_mass_com_wiring,
)
