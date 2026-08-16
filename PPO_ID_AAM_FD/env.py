"""Isaac Lab environment entry point for the online AAM ID-FD pipeline."""

from __future__ import annotations

from PPO_ID_FD.env import Ppo41IDFDEnv


class PpoIDAAMFDEnv(Ppo41IDFDEnv):
    """PPO4_1 task with online causal AAM and latent-conditioned ID adapter."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        if not bool(getattr(cfg, "online_aam_enabled", False)):
            raise ValueError("PPO_ID_AAM_FD requires online_aam_enabled=True")
        if not bool(getattr(cfg, "online_aam_policy_observation", False)):
            raise ValueError("PPO_ID_AAM_FD requires z_hat in the policy observation")
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def _expected_observation_dim(self) -> int:
        # The shared sole observation remains intact; Stage 31 appends the
        # two-dimensional causal z_arm_hat after it.
        return super()._expected_observation_dim() + 2
