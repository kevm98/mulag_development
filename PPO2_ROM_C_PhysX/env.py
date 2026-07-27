"""Robust ROM-B task extended with causal online AAM observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from PPO2_ROM_B_PhysX.env import PPO2ROMBPolicyEnv
from PPO2_ROM_C_PhysX.runtime import (
    OnlineAAMRuntime,
)


class PPO2CAAMPolicyEnv(PPO2ROMBPolicyEnv):
    """ROM-B policy conditioned on five supported AAM estimates.

    Ground-truth factors exist only inside ``_command_cylinder_rom`` and are
    intentionally never concatenated into the policy observation.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        control_dt = max(float(self.step_dt), 1.0e-6)
        # PPO2ROMBPolicyEnv already installed the exact ROM-B runtime. Do not
        # replace it with the older seven-factor ROM-C plant family.
        self._aam_runtime = OnlineAAMRuntime(
            checkpoint=cfg.aam_checkpoint,
            num_envs=self.num_envs,
            device=self.device,
            control_dt_s=control_dt,
            sample_time_s=float(cfg.aam_sample_time_s),
            ema_beta=float(cfg.aam_ema_beta),
        )
        self._aam_physical_factors = torch.zeros(
            self.num_envs, 5, device=self.device
        )
        self._aam_policy_factors = torch.zeros_like(self._aam_physical_factors)
        self._aam_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._aam_last_observation_step = -1
        print(
            "[PPO2-ROM-C] obs=47 action=4 | hidden ROM-B factors -> "
            "online AAM estimates -> policy; privileged factors disabled",
            flush=True,
        )

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        base = observations["policy"]
        # Isaac may request observations while the parent constructor is still
        # running. Preserve the 47-D contract with neutral estimates.
        if not hasattr(self, "_aam_runtime"):
            estimates = torch.zeros(
                base.shape[0], 5, dtype=base.dtype, device=base.device
            )
        elif int(self.common_step_counter) != self._aam_last_observation_step:
            piston_areas = torch.tensor(
                [0.2513, 0.2513, 0.1256, 0.1256],
                dtype=torch.float32, device=self.device,
            ).reshape(1, 4)
            force = self._dP * piston_areas
            physical, estimates, ready = self._aam_runtime.observe(
                valve=self._current_valve_cmd,
                extension=self._cylinder_x_current,
                velocity=self._cylinder_xdot_current,
                dP=self._dP,
                force=force,
            )
            self._aam_physical_factors = physical
            self._aam_policy_factors = estimates
            self._aam_ready = ready
            self._aam_last_observation_step = int(self.common_step_counter)
        else:
            estimates = self._aam_policy_factors
        observations["policy"] = torch.cat((base, estimates), dim=-1)
        if observations["policy"].shape[-1] != 47:
            raise RuntimeError(
                f"PPO2_C observation must be 47-D, got {observations['policy'].shape}"
            )
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        if hasattr(self, "_aam_ready"):
            self._last_reward_terms.update({
                "aam/ready_fraction": self._aam_ready.float().mean(),
                "aam/mean_abs_estimate": self._aam_policy_factors.abs().mean(),
            })
        return reward

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if hasattr(self, "_aam_runtime"):
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(
                    env_ids, dtype=torch.long, device=self.device
                )
            self._aam_runtime.reset(env_ids)
            self._aam_physical_factors[env_ids] = 0.0
            self._aam_policy_factors[env_ids] = 0.0
            self._aam_ready[env_ids] = False
