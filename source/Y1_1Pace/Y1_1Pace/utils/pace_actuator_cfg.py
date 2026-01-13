# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.actuators import DCMotorCfg
from Y1_1Pace.utils import pace_actuator


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    """Configuration for Pace DC Motor actuator model.

    This class extends the base DCMotorCfg with Pace-specific parameters.
    """
    class_type: type = pace_actuator.PaceDCMotor
    encoder_bias: list[float] | float | None = 0.0
    tanh_scale: int | list[int] | None = 100  # Scale factor w for tanh(w * torque)
