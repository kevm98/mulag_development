

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

class JointPositionNormalizedProcessedActionToLimitsAction(JointPositionAction):
    """ convert normalized joint position actions to joint position actions within joint limits
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions

        lower_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        upper_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1]
        self._processed_actions = (actions + 1) * (upper_limits - lower_limits) / 2 + lower_limits