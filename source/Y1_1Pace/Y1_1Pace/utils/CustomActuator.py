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
    from .CustomActuator_cfg import CustomActuatorCfg


class CustomActuator(ImplicitActuator):
    """Pace Implicit Actuator model with encoder bias and first-order low-pass filter.

    The actuator models an implicit actuator model with encoder bias and first-order low-pass filter on torques.

    The software implementation is inspired by ImplicitActuator.
    """

    cfg: CustomActuatorCfg

    def __init__(self, cfg: CustomActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)):
            if len(cfg.encoder_bias) != self.num_joints:
                raise ValueError(
                    f"encoder_bias must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.encoder_bias)}: {cfg.encoder_bias}"
                )
        self.encoder_bias = torch.tensor(cfg.encoder_bias, device=self._device).unsqueeze(0).repeat(self._num_envs, 1)


        if cfg.cutoff_frequency is None:
            self._tau = None
        else:
            if isinstance(cfg.cutoff_frequency, (list, tuple)):
                if len(cfg.cutoff_frequency) != self.num_joints:
                    raise ValueError(
                        f"cutoff_frequency must have {self.num_joints} elements (one per joint), "
                        f"but got {len(cfg.cutoff_frequency)}: {cfg.cutoff_frequency}"
                    )
            cutoff_freq = torch.tensor(cfg.cutoff_frequency, device=self._device)
            self._tau = 1.0 / (2.0 * 3.141592653589793 * cutoff_freq)  # Shape: (num_joints,)
        
        self._dt = None
        
        # Filtered torque state (initialized to zero)
        self._filtered_torques = torch.zeros(self._num_envs, self.num_joints, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # Reset filtered torques to zero
        self._filtered_torques[env_ids] = 0.0

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        
        if self._tau is not None:
            if self._dt is None:
                if hasattr(self, "_scene") and hasattr(self._scene, "sim") and hasattr(self._scene.sim, "cfg"):
                    self._dt = self._scene.sim.cfg.dt
                else:
                    self._dt = 0.002  
                self._alpha = self._dt / (self._dt + self._tau)
            
            alpha_expanded = self._alpha.unsqueeze(0)  
            self._filtered_torques = alpha_expanded * control_action_sim.joint_efforts + (1.0 - alpha_expanded) * self._filtered_torques
            control_action_sim.joint_efforts = self._filtered_torques
        
        return control_action_sim
