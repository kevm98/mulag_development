"""Isaac Lab environment entry point for the online AAM ID-FD pipeline."""

from __future__ import annotations

from PPO_ID_FD.env import Ppo41IDFDEnv


class PpoIDAAMFDEnv(Ppo41IDFDEnv):
    """PPO4_1 task with online causal AAM and latent-conditioned ID adapter."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        if not bool(getattr(cfg, "online_aam_enabled", False)):
            raise ValueError("PPO_AAM_FD requires online_aam_enabled=True")
        if not bool(getattr(cfg, "online_aam_policy_observation", False)):
            raise ValueError("PPO_AAM_FD requires z_hat in the policy observation")
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def _expected_observation_dim(self) -> int:
        # The shared sole observation remains intact; a context block is
        # appended after it. THREE mutually exclusive routes share that slot
        # and they are not the same width, so this must follow the same
        # derivation as verge_task.terrain_observations.observation_space_for_cfg
        # -- when the two disagree the env raises "observation_space=X but
        # sole_ik_hint=... requires Y" and the route cannot run at all, which
        # is what kept the privileged-e upper bound unusable.
        if bool(getattr(self.cfg, "online_aam_policy_privileged_e", False)):
            return super()._expected_observation_dim() + 6
        if bool(getattr(self.cfg, "online_aam_policy_rma_context", False)):
            return super()._expected_observation_dim() + 4
        if bool(getattr(self.cfg, "online_aam_policy_rif_context", False)):
            return super()._expected_observation_dim() + 2
        return super()._expected_observation_dim() + 2
