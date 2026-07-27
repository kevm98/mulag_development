from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IDFDPipelineSummary:
    observation: str = "lean EE YZ + cylinder + truck state"
    action: str = "residual PPO2 -> EE YZ DLS -> cylinder DLS action4"
    inverse_dynamics: str = "ID(target) - ID(zero) -> ValveCmd4"
    forward_dynamics: str = "rank-4 FD(valve) - FD(zero) -> delta_extension4"
    actuator: str = "PhysX set_joint_position_target on four cylinder joints"
    disabled_models: str = "ROM disabled, sequence disabled, no AAM/SD branch"


def pipeline_summary() -> IDFDPipelineSummary:
    return IDFDPipelineSummary()
