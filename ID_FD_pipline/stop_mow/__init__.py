"""Stop-and-mow state machine and reference management.

Provides a deterministic per-environment state machine that sequences:
RESET_SETTLE -> DEPLOY -> MOW_OUT -> MOW_RETURN -> (repeat or STOW) -> COMPLETE.
The truck is parked during all arm work phases.
"""

from .phases import StopMowPhase, StopMowState
from .state_machine import StopMowStateMachine
from .parking import ParkingBrake
from .reference_manager import ReferenceManager
from .ground_following import GroundFollower

__all__ = [
    "StopMowPhase",
    "StopMowState",
    "StopMowStateMachine",
    "ParkingBrake",
    "ReferenceManager",
    "GroundFollower",
]
