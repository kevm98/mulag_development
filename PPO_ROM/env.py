"""Final PPO-ROM environment with the bounded task reward."""

from __future__ import annotations

import torch

from PPO_ROM.terrain_env import TerrainC4AuthorityEnv

from PPO_ROM.rewards import RewardWeights, compute_reward
from verge_task.vehicle_speed import episode_path_active_mask


class Ppo31SoleEnv(TerrainC4AuthorityEnv):
    """Verge mowing with the ten-term bounded reward."""

    def _reward_weights(self) -> RewardWeights:
        """Weights, overridable from the cfg without inventing config fields."""
        cached = getattr(self, "_ppo31_weights", None)
        if cached is not None:
            return cached
        overrides = {}
        for field in RewardWeights.__dataclass_fields__:
            value = getattr(self.cfg, f"ppo31_{field}", None)
            if value is not None:
                overrides[field] = float(value)
        weights = RewardWeights(**overrides)
        self._ppo31_weights = weights
        return weights

    def _get_rewards(self) -> torch.Tensor:
        self._update_contact_safety()
        extension, _ = self._refresh_cylinder_state()
        headroom_low, headroom_high = self._cylinder_headroom(extension)

        terms = compute_reward(
            cut_height_error=self.cut_height_error(),
            lateral_error=self.verge_lateral_error(),
            head_down_alignment=self._measure_head_down_alignment(),
            cutter_contact=self.cutter_ground_contact(),
            applied_action=self._sole_applied,
            previous_applied_action=self._prev_sole_applied,
            tool_acceleration_norm=torch.linalg.norm(
                self._measured_tool_acceleration_truck, dim=-1
            ),
            contact_force_n=self.cutter_terrain_contact_force_n(),
            cylinder_headroom_low_m=headroom_low,
            cylinder_headroom_high_m=headroom_high,
            hard_impact=self._hard_impact_event,
            weights=self._reward_weights(),
        )

        reward = terms.total

        active = self._scored_active_mask()
        reward = torch.where(active, reward, torch.zeros_like(reward))

        invalid = ~torch.isfinite(reward)
        if bool(invalid.any()):
            reward = torch.where(invalid, torch.full_like(reward, -25.0), reward)

        self._ppo31_last_terms = terms
        self._publish_log(terms, reward)
        return reward

    def _scored_active_mask(self) -> torch.Tensor:
        """Return replicas that are inside the scored part of their path.

        Vectorized episodes reset together because physical deployment is
        synchronized.  With episode-level speed randomization, faster replicas
        therefore spend an absorbing tail parked at the endpoint while the
        slowest replica finishes.  Those tail steps must contribute exactly
        zero task reward or the speed distribution would change the objective.
        """
        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for attribute in ("_scored_active", "_episode_active", "_active_mask"):
            mask = getattr(self, attribute, None)
            if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
                active &= mask
                break
        preposition = getattr(self, "_preposition_counter", None)
        settle = getattr(self, "_settle_counter", None)
        complete = getattr(self, "_truck_path_complete", None)
        if all(isinstance(value, torch.Tensor) for value in (preposition, settle, complete)):
            active = episode_path_active_mask(
                preposition,
                settle,
                complete,
                base_mask=active,
            )
        return active

    def _publish_log(self, terms, reward: torch.Tensor) -> None:
        """Standard Isaac Lab log namespaces, read by the rsl_rl runner.

        The parent populates `extras["log"]` at the end of its own
        `_get_rewards`; overriding that method therefore silences ALL logging
        unless it is re-published here. Namespacing follows Isaac Lab's
        manager-based convention (`Episode_Reward/<term>`), which is what the
        teacher_student extension uses, so tensorboard groups the terms the way
        a reader expects.

        Every term is logged individually and by name. The old stack logged an
        aggregate plus a partial subset -- the two largest lateral terms were
        not logged at all, which is part of why a reward term that opposed the
        task went unnoticed for thirteen campaigns.
        """
        log = self.extras.setdefault("log", {})
        for name, value in terms.as_dict().items():
            log[f"Episode_Reward/{name}"] = value.mean()
        log["Episode_Reward/total"] = reward.mean()

        log["Metrics/lateral_error_m"] = self.verge_lateral_error().abs().mean()
        log["Metrics/cut_height_error_m"] = self.cut_height_error().abs().mean()
        log["Metrics/ground_contact_fraction"] = (
            self.cutter_ground_contact().float().mean()
        )
        log["Metrics/head_alignment"] = self._measure_head_down_alignment().mean()

        # Keep the contact-safety namespace that the shared terrain evaluator
        # and smoke gate consume.  PPO-ROM owns this reward override, so the
        # parent logger cannot populate these fields for us.  The airborne-gap
        # reward is not a separate ROM term; expose the measured gap as a
        # diagnostic and use zero for its legacy reward slot.
        active = self._scored_active_mask()
        if bool(active.any()):
            log["contact/acquired_fraction"] = self._contact_acquired[active].float().mean()
            log["contact/loss_event_fraction"] = self._contact_loss_event[active].float().mean()
            log["contact/max_airborne_gap_m"] = self._airborne_gap_m[active].max()
            log["reward/contact_total"] = (
                terms.ground_contact[active] + terms.contact_loss[active]
            ).mean()
            log["reward/contact_loss"] = terms.contact_loss[active].mean()
            log["reward/contact_airborne_gap"] = torch.zeros(
                (), device=reward.device, dtype=reward.dtype
            )
