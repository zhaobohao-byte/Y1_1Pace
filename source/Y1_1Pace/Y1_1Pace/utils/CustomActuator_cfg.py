# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.actuators import ImplicitActuatorCfg
from Y1_1Pace.utils.CustomActuator import CustomActuator


@configclass
class CustomActuatorCfg(ImplicitActuatorCfg):
    """Configuration for Pace Implicit Actuator model.

    This class extends the base ImplicitActuatorCfg with Pace-specific parameters.
    """
    class_type: type = CustomActuator
    encoder_bias: list[float] | float | None = 0.0
    cutoff_frequency: list[float] | float | None = 500.0 # Cutoff frequency for first-order low-pass filter (Hz), default 1kHz
