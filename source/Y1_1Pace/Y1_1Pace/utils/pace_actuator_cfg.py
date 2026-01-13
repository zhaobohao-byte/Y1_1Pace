# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.actuators import DCMotorCfg
from Y1_1Pace.utils import pace_actuator


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    """Configuration for Pace DC Motor actuator model with delay."""
    class_type: type = pace_actuator.PaceDCMotor
    max_delay: int = 0
    """Maximum delay in simulation steps."""
