from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from .actions import JointPositionNormalizedProcessedActionToLimitsAction

from isaaclab.utils import configclass

@configclass
class JointPositionNormalizedToLimitsActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = JointPositionNormalizedProcessedActionToLimitsAction