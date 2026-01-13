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
    # only for type checking
    from .pace_actuator_cfg import PaceDCMotorCfg


class PaceDCMotor(DCMotor):
    """Pace DC Motor actuator model with encoder bias, tanh torque scaling, and action delay.

    The actuator models a DC motor whose controller receives joint positions in the encoder
    frame by adding a per-joint encoder bias to the true joint positions. In other words,
    the controller operates on biased (encoder) positions rather than the true joint positions.

    The torque command computed by the PD controller is processed through a tanh function
    with a configurable scale factor: tanh(w * torque), where w is the tanh_scale parameter.
    This provides smooth saturation of torque commands.
    """

    cfg: PaceDCMotorCfg

    def __init__(self, cfg: PaceDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)):
            if len(cfg.encoder_bias) != self.num_joints:
                raise ValueError(
                    f"encoder_bias must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.encoder_bias)}: {cfg.encoder_bias}"
                )
        self.encoder_bias = torch.tensor(cfg.encoder_bias, device=self._device).unsqueeze(0).repeat(self._num_envs, 1)
        self.torques_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))

        # Initialize tanh scale factor
        if isinstance(cfg.tanh_scale, (list, tuple)):
            if len(cfg.tanh_scale) != self.num_joints:
                raise ValueError(
                    f"tanh_scale must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.tanh_scale)}: {cfg.tanh_scale}"
                )
            self.tanh_scale = torch.tensor(cfg.tanh_scale, dtype=torch.float32, device=self._device).unsqueeze(0).repeat(self._num_envs, 1)
        else:
            self.tanh_scale = torch.full((self._num_envs, self.num_joints), float(cfg.tanh_scale), dtype=torch.float32, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # reset buffers
        self.torques_delay_buffer.reset(env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(delay, env_ids)

    def update_tanh_scale(self, tanh_scale: torch.Tensor):
        self.tanh_scale = tanh_scale

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:

        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        saturated_torques = torch.tanh(self.tanh_scale * control_action_sim.joint_efforts)
        control_action_sim.joint_efforts = self.torques_delay_buffer.compute(saturated_torques)

        return control_action_sim
