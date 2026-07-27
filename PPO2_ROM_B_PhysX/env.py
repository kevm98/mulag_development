"""Factor-blind PPO2-ROM_B environment."""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from PPO2_ROM_B_PhysX.rewards import compute_rom_b_safety_reward
from PPO2_ROM_B_PhysX.runtime import (
    ROMBCommandCylinderROMRuntime,
    ROMBFactorConfig,
    SUPPORTED_FACTOR_NAMES,
)
from PPO2_ROM_B_PhysX.runtime.plant_observer import (
    PLANT_LAG_OBS_DIM,
    PlantLagObserver,
)
from PPO2_ROM_PhysX.policy.env import ResidualPolicyEnv


# Observation slice holding `corrected_cylinder_goal - measured_extension`
# (see PPO2_ROM_PhysX/policy/observations.py layout comment).
_CYLINDER_GOAL_ERROR_SLICE = slice(16, 20)

# Privileged oracle. The runtime exposes exactly the five supported factors.
PRIVILEGED_FACTOR_NAMES = (
    "e_comp", "d_seal_1", "d_seal_2", "d_seal_3", "d_seal_4"
)
PRIVILEGED_OBS_DIM = len(PRIVILEGED_FACTOR_NAMES)


class PPO2ROMBPolicyEnv(ResidualPolicyEnv):
    """PPO2-A task with hidden, reset-fixed ROM-B hydraulic factors."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._rom_b_truck_anchor_body_pose = self.robot.data.body_link_state_w[
            :, self._truck_base_body_id(), :7
        ].detach().clone()
        self._command_cylinder_rom = ROMBCommandCylinderROMRuntime(
            checkpoint=cfg.command_rom_checkpoint,
            num_envs=self.num_envs,
            device=self.device,
            control_dt_s=max(float(self.step_dt), 1.0e-6),
            factor_cfg=ROMBFactorConfig(
                mode=str(cfg.rom_b_mode),
                e_comp_range=tuple(cfg.rom_b_e_comp_range),
                seal_range=tuple(cfg.rom_b_seal_range),
                health_scale=float(cfg.rom_b_health_scale),
                randomization_scale=float(cfg.rom_b_randomization_scale),
            ),
        )
        self._plant_observer: PlantLagObserver | None = None
        self._last_plant_features: torch.Tensor | None = None
        if bool(getattr(cfg, "rom_b_plant_observer", False)):
            self._plant_observer = PlantLagObserver(
                num_envs=self.num_envs,
                device=self.device,
                control_dt_s=max(float(self.step_dt), 1.0e-6),
                fast_tau_s=float(getattr(cfg, "rom_b_plant_observer_fast_tau_s", 0.25)),
                slow_tau_s=float(getattr(cfg, "rom_b_plant_observer_slow_tau_s", 2.0)),
            )
        self._privileged_factors = bool(
            getattr(cfg, "rom_b_privileged_factors", False)
        )
        observed = (
            42
            + (PLANT_LAG_OBS_DIM if self._plant_observer else 0)
            + (PRIVILEGED_OBS_DIM if self._privileged_factors else 0)
        )
        print(
            f"[PPO2-ROM-B] obs={observed} action=4 | randomized factors="
            f"{list(SUPPORTED_FACTOR_NAMES)} | actor/critic "
            + ("FACTOR-PRIVILEGED" if self._privileged_factors else "factor-blind")
            + (
                f" | plant-RESPONSE observer ON (+{PLANT_LAG_OBS_DIM} dims: a temporal"
                " summary of commanded-vs-achieved motion, NOT a factor estimate)"
                if self._plant_observer else ""
            ),
            flush=True,
        )
        if self._privileged_factors:
            print(
                "[PPO2-ROM-B] *** PRIVILEGED ORACLE: the true factors "
                f"{list(PRIVILEGED_FACTOR_NAMES)} are in the observation. This "
                "run is a DIAGNOSTIC UPPER BOUND and can never be reported as a "
                "ROM-B robustness result. ***",
                flush=True,
            )

    @torch.no_grad()
    def _hold_wheels(self) -> None:
        """Use the shared hold, then hard-fix wheel positions for GUI playback."""
        super()._hold_wheels()
        if not bool(getattr(self.cfg, "truck_full_pose_fixture", False)):
            return
        if self._wheel_init_pos is not None:
            self.robot.write_joint_position_to_sim(
                self._wheel_init_pos, joint_ids=self._wheel_joint_ids
            )

    @torch.no_grad()
    def _constrain_truck_xy(self) -> None:
        """Optionally fix the actual chassis while leaving arm joints free."""
        anchor = getattr(self, "_rom_b_truck_anchor_body_pose", None)
        if not bool(getattr(self.cfg, "truck_full_pose_fixture", False)) or anchor is None:
            super()._constrain_truck_xy()
            return
        base = self.robot.data.body_link_state_w[:, self._truck_base_body_id(), :7]
        delta_q = quat_mul(anchor[:, 3:7], quat_conjugate(base[:, 3:7]))
        root_pose = self.robot.data.root_state_w[:, :7].detach().clone()
        root_from_base = root_pose[:, :3] - base[:, :3]
        root_pose[:, :3] = anchor[:, :3] + quat_apply(delta_q, root_from_base)
        root_pose[:, 3:7] = quat_mul(delta_q, root_pose[:, 3:7])
        self.robot.write_root_pose_to_sim(root_pose)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((self.num_envs, 6), device=self.device)
        )
        self._truck_drift_xy.zero_()
        self._truck_post_constraint_drift_xy.zero_()

    def _append_privileged_factors(self, obs: dict) -> dict:
        """Append the true hidden factors. DIAGNOSTIC ONLY -- never certifiable.

        Establishes the upper bound on adaptation: an online estimator can only
        be worse than the truth, so this policy's margin over factor-blind ROM-B
        ceilings what AAM can achieve.
        """

        factors = self._command_cylinder_rom.true_factors
        obs["policy"] = torch.cat((obs["policy"], factors), dim=-1)
        if "critic" in obs:
            obs["critic"] = torch.cat((obs["critic"], factors), dim=-1)
        return obs

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        if self._plant_observer is None:
            if self._privileged_factors:
                obs = self._append_privileged_factors(obs)
            return obs
        policy = obs["policy"]
        active = (self._preposition_counter <= 0) & (self._settle_counter <= 0)
        features = self._plant_observer.update(
            policy[:, _CYLINDER_GOAL_ERROR_SLICE], active
        )
        # Ablation interventions. "zero" reproduces observer-off exactly for a
        # widened policy; "shuffle" keeps the marginal feature distribution but
        # destroys the environment-to-plant correspondence, separating "the
        # policy uses these features" from "the policy uses them informatively".
        mode = str(getattr(self.cfg, "rom_b_plant_observer_mode", "on")).lower()
        if mode == "zero":
            features = torch.zeros_like(features)
        elif mode == "shuffle":
            features = features[
                torch.randperm(features.shape[0], device=features.device)
            ]
        elif mode != "on":
            raise ValueError(
                f"rom_b_plant_observer_mode must be on|zero|shuffle, got {mode!r}"
            )
        self._last_plant_features = features
        obs["policy"] = torch.cat((policy, features), dim=-1)
        if "critic" in obs:
            obs["critic"] = torch.cat((obs["critic"], features), dim=-1)
        if self._privileged_factors:
            obs = self._append_privileged_factors(obs)
        return obs

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        if self._plant_observer is not None:
            ids = (
                None
                if env_ids is None
                else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            )
            self._plant_observer.reset(ids)

    def set_rom_b_randomization_scale(self, scale: float) -> None:
        if not isinstance(self._command_cylinder_rom, ROMBCommandCylinderROMRuntime):
            raise RuntimeError("ROM-B runtime is not initialized")
        self._command_cylinder_rom.set_randomization_scale(scale)
        self.cfg.rom_b_randomization_scale = float(scale)
        print(
            f"[PPO2-ROM-B] curriculum randomization_scale={float(scale):.2f}; "
            "all environments resampled",
            flush=True,
        )

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        cfg = self.cfg
        head_alignment = self._measure_head_down_alignment()
        clearance_error = self._clearance_signal()
        acceleration = torch.linalg.norm(
            self._measured_tool_acceleration_truck, dim=-1
        )
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids]
        limit_margin = float(getattr(cfg, "cylinder_limit_margin_m", 0.002))
        cylinder_limit_violation = (
            (cylinder_q < limits[..., 0] - limit_margin)
            | (cylinder_q > limits[..., 1] + limit_margin)
        ).any(dim=-1)
        truck_violation = (
            self._truck_post_constraint_drift_xy
            > float(getattr(cfg, "truck_drift_termination_m", 0.02))
        ) | (
            self._truck_drift_xy
            > float(getattr(cfg, "truck_raw_reaction_termination_m", 0.05))
        )
        safety = compute_rom_b_safety_reward(
            head_alignment=head_alignment,
            clearance_error_m=clearance_error,
            acceleration_mps2=acceleration,
            additional_safety_violation=cylinder_limit_violation | truck_violation,
            truck_reaction_m=self._truck_drift_xy,
            truck_reaction_soft_m=float(
                getattr(cfg, "rom_b_truck_reaction_soft_m", 0.035)
            ),
            truck_reaction_hard_m=float(
                getattr(cfg, "rom_b_truck_reaction_hard_m", 0.050)
            ),
            truck_reaction_margin_weight=float(
                getattr(cfg, "rom_b_truck_reaction_margin_weight", 0.0)
            ),
            head_soft=float(cfg.rom_b_head_soft_alignment),
            head_hard=float(cfg.rom_b_head_hard_alignment),
            clearance_soft_m=float(cfg.rom_b_clearance_soft_m),
            clearance_hard_m=float(cfg.rom_b_clearance_hard_m),
            acceleration_soft_mps2=float(cfg.rom_b_acceleration_soft_mps2),
            acceleration_hard_mps2=float(cfg.rom_b_acceleration_hard_mps2),
            head_margin_weight=float(cfg.rom_b_head_margin_weight),
            clearance_margin_weight=float(cfg.rom_b_clearance_margin_weight),
            acceleration_margin_weight=float(cfg.rom_b_acceleration_margin_weight),
            hard_safety_event_weight=float(cfg.rom_b_hard_safety_event_weight),
            acceleration_event_weight=float(cfg.rom_b_acceleration_event_weight),
            clearance_in_hard_safety_event=bool(
                getattr(cfg, "rom_b_clearance_in_hard_safety_event", True)
            ),
        )
        active = (self._preposition_counter <= 0) & (self._settle_counter <= 0)
        reward = reward + torch.where(
            active, safety.reward, torch.zeros_like(safety.reward)
        )
        zero = torch.zeros((), device=self.device)
        clearance_violation = (
            torch.abs(clearance_error) > float(cfg.rom_b_clearance_hard_m)
        )
        head_violation = head_alignment < float(cfg.rom_b_head_hard_alignment)
        self._last_reward_terms.update({
            # Per-step occupancy by cause. A per-rollout termination fraction does
            # not determine per-step violation occupancy, so these are logged
            # rather than inferred.
            "rom_b_safety/clearance_violation_occupancy": torch.mean(
                clearance_violation[active].float()
            ) if torch.any(active) else zero,
            "rom_b_safety/head_violation_occupancy": torch.mean(
                head_violation[active].float()
            ) if torch.any(active) else zero,
            "rom_b_safety/mean_hard_event_penalty": torch.mean(
                safety.hard_safety_penalty[active]
            ) if torch.any(active) else zero,
            "rom_b_safety/truck_reaction_margin_penalty": torch.mean(
                safety.truck_reaction_margin_penalty[active]
            ) if torch.any(active) else zero,
            "rom_b_safety/truck_reaction_violation_rate": torch.mean(
                (
                    self._truck_drift_xy
                    > float(getattr(cfg, "rom_b_truck_reaction_hard_m", 0.050))
                )[active].float()
            ) if torch.any(active) else zero,
            "rom_b_safety/hard_violation_rate": torch.mean(
                safety.hard_safety_violation[active].float()
            ) if torch.any(active) else zero,
            "rom_b_safety/acceleration_violation_rate": torch.mean(
                safety.acceleration_violation[active].float()
            ) if torch.any(active) else torch.zeros((), device=self.device),
            "rom_b_safety/mean_penalty": torch.mean(safety.reward[active])
            if torch.any(active) else torch.zeros((), device=self.device),
        })
        if isinstance(self._command_cylinder_rom, ROMBCommandCylinderROMRuntime):
            self._last_reward_terms.update(
                self._command_cylinder_rom.factor_log_scalars()
            )
            self.extras.setdefault("log", {}).update(self._last_reward_terms)
        return reward
