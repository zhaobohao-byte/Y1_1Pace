# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.actuators import DCMotorCfg
from Y1_1Pace.utils import customed_actuator


@configclass
class CustomedDCMotorCfg(DCMotorCfg):
    """Configuration for Pace DC Motor actuator model with first-order low-pass filter.

    This class extends the base DCMotorCfg with a low-pass filter for torque smoothing.
    The filter helps simulate the dynamic response characteristics of real actuators.
    """
    class_type: type = customed_actuator.CustomedDCMotor
    cutoff_frequency: int | float = 100
    """Cutoff frequency for the first-order low-pass filter in Hz. Default is 100 Hz."""
