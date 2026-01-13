# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.actuators import DCMotorCfg
from Y1_1Pace.utils import pace_actuator


@configclass
class PaceDCMotorCfg(DCMotorCfg):

    class_type: type = pace_actuator.PaceDCMotor
    encoder_bias: list[float] | float | None = 0.0
    max_delay: int | None = 0
    tanh_scale: int | list[int] | None = 100
