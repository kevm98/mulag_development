from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import Optional

from isaaclab.managers.recorder_manager import RecorderTerm


class PostStepNormalizedJointPositionActionsRecorder(RecorderTerm):
    """Recorder term that records the proceseed joint position actions which send to the env,
       then the processed action will be normalized to [-1, 1] based on the joint limits then recorded.
    """

    def record_post_step(self):
        processed_actions: Optional[torch.Tensor] = None

        # Loop through active terms and concatenate their processed actions
        for term_name in self._env.action_manager.active_terms:
            term_actions = self._env.action_manager.get_term(term_name).processed_actions.clone()
            if processed_actions is None:
                processed_actions = term_actions
            else:
                processed_actions = torch.cat([processed_actions, term_actions], dim=-1)

        return "actions", self.normalize_actions(processed_actions)
    

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize the actions to [-1, 1] based on joint limits.

        Args:
            actions (torch.Tensor): The processed joint position actions.

        Returns:
            torch.Tensor: The normalized joint position actions.
        """
        joint_ids = self._env.action_manager.get_term("joint_pos")._joint_ids  # type: ignore
        lower_limits = self._env.scene["robot"].data.soft_joint_pos_limits[:, joint_ids, 0]
        upper_limits = self._env.scene["robot"].data.soft_joint_pos_limits[:, joint_ids, 1]

        # Normalize actions to [-1, 1]
        normalized_actions = 2 * (actions - lower_limits) / (upper_limits - lower_limits) - 1
        normalized_actions = torch.clamp(normalized_actions, -1.0, 1.0)

        return normalized_actions