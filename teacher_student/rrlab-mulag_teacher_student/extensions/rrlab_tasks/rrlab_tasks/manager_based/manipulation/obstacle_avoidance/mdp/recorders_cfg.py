from __future__ import annotations
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.recorders.recorders_cfg import InitialStateRecorderCfg, PostStepStatesRecorderCfg, PostStepProcessedActionsRecorderCfg


from .recorders import PostStepNormalizedJointPositionActionsRecorder, PreStepStudentObservationsRecorder, ExperimentInfoRecorder

@configclass
class PostStepNormalizedJointPositionActionsRecorderCfg(RecorderTermCfg):
    
    class_type: type[RecorderTerm] = PostStepNormalizedJointPositionActionsRecorder

@configclass
class PreStepStudentObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = PreStepStudentObservationsRecorder


@configclass
class MulagStudentObsTeacherActRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PostStepNormalizedJointPositionActionsRecorderCfg()
    record_pre_step_flat_policy_observations = PreStepStudentObservationsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()


@configclass
class ExperimentInfoRecorderTermCfg(RecorderTermCfg):
    
    class_type: type[RecorderTerm] = ExperimentInfoRecorder

@configclass
class ExperimentInfoRecorderCfg(RecorderManagerBaseCfg):
    

    experiment_info=ExperimentInfoRecorderTermCfg()
    dataset_export_dir_path="logs/mulag_eval/experiment_info"
    dataset_filename = "mulag_experiment_info"


