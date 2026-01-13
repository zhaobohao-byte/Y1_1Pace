# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.actuators import DCMotorCfg
from Y1_1Pace.utils import customed_actuator


@configclass
class CustomedDCMotorCfg(DCMotorCfg):
    """DC Motor with tanh smoothing and delay buffer."""
    class_type: type = customed_actuator.CustomedDCMotor
    smoothing_coefficient: list[int] | int = 10.0
    """Smoothing coefficient (w) for tanh: tau_out = tanh(w * tau_in) / w."""
    max_delay: int = 0
    """Maximum delay in simulation steps."""
