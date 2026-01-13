# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators import DCMotor
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils import DelayBuffer

if TYPE_CHECKING:
    from .customed_actuator_cfg import CustomedDCMotorCfg


class CustomedDCMotor(DCMotor):
    """DC Motor with tanh smoothing and delay buffer.

    Torque: tau_out = tanh(w * tau_in) / w, then delayed.
    """

    cfg: CustomedDCMotorCfg

    def __init__(self, cfg: CustomedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # Smoothing coefficient
        if isinstance(cfg.smoothing_coefficient, (list, tuple)):
            if len(cfg.smoothing_coefficient) != self.num_joints:
                raise ValueError(
                    f"smoothing_coefficient must have {self.num_joints} elements, "
                    f"got {len(cfg.smoothing_coefficient)}")
            smoothing_coeff = torch.tensor(cfg.smoothing_coefficient,
                                           device=self._device, dtype=torch.float32)
        else:
            smoothing_coeff = torch.full((self.num_joints,), cfg.smoothing_coefficient,
                                         device=self._device, dtype=torch.float32)

        self.smoothing_coefficients = smoothing_coeff

        # Delay buffer
        self.torques_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self.torques_delay_buffer.reset(env_ids)

    def update_smoothing_coefficient(self, smoothing_coefficient: float | list[float] | torch.Tensor):
        """Update smoothing coefficient (w)."""
        if isinstance(smoothing_coefficient, torch.Tensor):
            coeff = smoothing_coefficient.to(self._device)
        elif isinstance(smoothing_coefficient, (list, tuple)):
            if len(smoothing_coefficient) != self.num_joints:
                raise ValueError(f"Expected {self.num_joints} elements, got {len(smoothing_coefficient)}")
            coeff = torch.tensor(smoothing_coefficient, device=self._device, dtype=torch.float32)
        else:
            coeff = torch.full((self.num_joints,), smoothing_coefficient,
                               device=self._device, dtype=torch.float32)
        self.smoothing_coefficients = coeff

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        """Update delay for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(delay, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # DC motor PD control
        control_action_sim = super().compute(control_action, joint_pos, joint_vel)

        # Tanh smoothing: tau_out = tanh(w * tau_in) / w
        w = self.smoothing_coefficients.unsqueeze(0)
        tau_in = control_action_sim.joint_efforts
        smoothed_torques = torch.tanh(w * tau_in) / w

        # Apply delay
        control_action_sim.joint_efforts = self.torques_delay_buffer.compute(smoothed_torques)

        return control_action_sim
