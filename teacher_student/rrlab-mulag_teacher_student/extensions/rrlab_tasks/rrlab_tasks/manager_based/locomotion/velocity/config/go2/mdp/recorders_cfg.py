from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import recorders

@configclass
class PostStepNormalizedJointPositionActionsRecorderCfg(RecorderTermCfg):
    
    class_type: type[RecorderTerm] = recorders.PostStepNormalizedJointPositionActionsRecorder