# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.actuators import ImplicitActuatorCfg
from Y1_1Pace.utils import pace_implicitActuator


@configclass
class PaceImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for Pace Implicit Actuator model.

    This class extends the base ImplicitActuatorCfg with Pace-specific parameters.
    """
    class_type: type = pace_implicitActuator.PaceImplicitActuator
