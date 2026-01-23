# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators import ImplicitActuator
from isaaclab.utils.types import ArticulationActions
if TYPE_CHECKING:
    # only for type checking
    from .pace_implicitActuator_cfg import PaceImplicitActuatorCfg


class PaceImplicitActuator(ImplicitActuator):
    """Pace Implicit Actuator model.

    The actuator models an implicit actuator model.

    The software implementation is inspired by ImplicitActuator.
    """

    cfg: PaceImplicitActuatorCfg

    def __init__(self, cfg: PaceImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute actuator model
        control_action_sim = super().compute(control_action, joint_pos, joint_vel)
        return control_action_sim
