# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.actuators import DCMotorCfg
from Y1_1Pace.utils import customed_actuator


@configclass
class CustomedDCMotorCfg(DCMotorCfg):
    """Configuration for Pace DC Motor actuator model with tanh-based torque smoothing.

    This class extends the base DCMotorCfg with a tanh function for torque smoothing.
    The torque output is computed as: tau_out = tanh(w * tau_in) / w
    where w is the smoothing coefficient.
    """
    class_type: type = customed_actuator.CustomedDCMotor
    smoothing_coefficient: list[float] | float = 1.0
    """Smoothing coefficient (w) for tanh function: tau_out = tanh(w * tau_in) / w.
    Can be a single value (applied to all joints) or a list of values (one per joint).
    Larger values (w > 1) provide less smoothing, smaller values (w < 1) provide more smoothing.
    Default is 1.0 (no smoothing)."""
