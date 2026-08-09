"""PPO3_1 environment: PPO3's physics, PPO3_1's reward.

Inherits `TerrainC4AuthorityEnv`, so a run gets BOTH measured findings at once:
the c4 command authority (larger lead cap and action scale on the head-pitch
cylinder) and the rebuilt nine-term reward.

Measured 2026-08-05 with `scripts/probe_c4_authority.py`, zero action, 16x480:
c4 runs at 21.6 kN mean against a 320 kN effort limit -- 6.8 % utilisation,
saturating 0.026 % of steps. So c4 is NOT force-limited and raising
`effort_limit_sim` would achieve nothing; the binding constraint is the lead cap
and action scale, which is exactly what the inherited class raises. (c1 is the
loaded axis: 427 kN of 800 kN mean, saturating 1.3 % of steps.)

Deliberately a thin subclass. The plant was never the problem -- the ROM,
the terrain generator, the contact sensing and the observation contract are all
sound and heavily validated. What failed was the reward (53 weighted terms,
three of which fought the task), the acceptance criteria (26 gates on one mesh)
and the training regime (warm-start fine-tuning at 5e-7).

So this class changes exactly one thing: `_get_rewards`. Everything else --
observations, terrain, resets, terminations, logging of physical metrics --
is inherited unchanged, which also means every PPO3 diagnostic script still
works against it.
"""

from __future__ import annotations

import torch

from PPO_ROM.terrain_env import TerrainC4AuthorityEnv

from PPO_ROM.rewards import RewardWeights, compute_reward


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
        # Reuse the parent's state machine so reward, termination and metrics
        # all read the SAME transition -- the parent caches these, so this is
        # not a second physics query.
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
            weights=self._reward_weights(),
        )

        reward = terms.total

        # A warm-up step must not be paid, exactly as in the parent.
        active = self._scored_active_mask()
        reward = torch.where(active, reward, torch.zeros_like(reward))

        # Non-finite physics is a failure, not a large negative number that the
        # critic then has to model. Bounded, like every other term.
        invalid = ~torch.isfinite(reward)
        if bool(invalid.any()):
            reward = torch.where(invalid, torch.full_like(reward, -25.0), reward)

        self._ppo31_last_terms = terms
        self._publish_log(terms, reward)
        return reward

    def _scored_active_mask(self) -> torch.Tensor:
        """The parent's notion of 'this step counts', by whatever name it uses."""
        for attribute in ("_scored_active", "_episode_active", "_active_mask"):
            mask = getattr(self, attribute, None)
            if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
                return mask
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

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

        # Task metrics, in the units the gates are written in, so a training
        # curve can be read against acceptance without a separate evaluation.
        log["Metrics/lateral_error_m"] = self.verge_lateral_error().abs().mean()
        log["Metrics/cut_height_error_m"] = self.cut_height_error().abs().mean()
        log["Metrics/ground_contact_fraction"] = (
            self.cutter_ground_contact().float().mean()
        )
        log["Metrics/head_alignment"] = self._measure_head_down_alignment().mean()
