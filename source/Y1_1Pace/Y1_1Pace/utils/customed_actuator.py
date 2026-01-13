# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators import DCMotor
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    # only for type checking
    from .customed_actuator_cfg import CustomedDCMotorCfg


class CustomedDCMotor(DCMotor):
    """Customized DC Motor actuator model with tanh-based torque smoothing.

    The actuator models a DC motor with a tanh function applied to the torque output.
    This provides smooth saturation characteristics and more realistic behavior.

    The torque output is computed as:
        tau_out = tanh(w * tau_in) / w

    where:
        - tau_in: input torque from PD controller
        - w: smoothing coefficient (per joint)
        - tau_out: output torque after smoothing

    When w = 1, the function approximates linear behavior for small torques.
    Smaller w values provide more smoothing and earlier saturation.
    Larger w values provide less smoothing and later saturation.
    """

    cfg: CustomedDCMotorCfg

    def __init__(self, cfg: CustomedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # Handle smoothing_coefficient as either a single value or a list
        if isinstance(cfg.smoothing_coefficient, (list, tuple)):
            if len(cfg.smoothing_coefficient) != self.num_joints:
                raise ValueError(
                    f"smoothing_coefficient must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.smoothing_coefficient)}: {cfg.smoothing_coefficient}"
                )
            smoothing_coeff_tensor = torch.tensor(cfg.smoothing_coefficient,
                                                  device=self._device, dtype=torch.float32)
        else:
            # Single value: apply to all joints
            smoothing_coeff_tensor = torch.full((self.num_joints,), cfg.smoothing_coefficient,
                                                device=self._device, dtype=torch.float32)

        # Store smoothing coefficients (w) for each joint
        self.smoothing_coefficients = smoothing_coeff_tensor  # shape: (num_joints,)

    def reset(self, env_ids: Sequence[int]):
        """Reset the actuator state for specified environments."""
        super().reset(env_ids)

    def update_smoothing_coefficient(self, smoothing_coefficient: float | list[float] | torch.Tensor):
        """Update the smoothing coefficient for the tanh function.

        Args:
            smoothing_coefficient: New smoothing coefficient (w).
                Can be a single value (applied to all joints),
                a list of values (one per joint),
                or a torch.Tensor of shape (num_joints,)
        """
        # Convert to tensor
        if isinstance(smoothing_coefficient, torch.Tensor):
            smoothing_coeff_tensor = smoothing_coefficient.to(self._device)
        elif isinstance(smoothing_coefficient, (list, tuple)):
            if len(smoothing_coefficient) != self.num_joints:
                raise ValueError(
                    f"smoothing_coefficient must have {self.num_joints} elements, "
                    f"but got {len(smoothing_coefficient)}"
                )
            smoothing_coeff_tensor = torch.tensor(smoothing_coefficient,
                                                  device=self._device, dtype=torch.float32)
        else:
            # Single value: apply to all joints
            smoothing_coeff_tensor = torch.full((self.num_joints,), smoothing_coefficient,
                                                device=self._device, dtype=torch.float32)

        self.smoothing_coefficients = smoothing_coeff_tensor

        print(f"[CustomedDCMotor] Updated smoothing coefficients to: {self.smoothing_coefficients.cpu().numpy()}")

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Compute actuator model (DC motor with PD control)
        control_action_sim = super().compute(control_action, joint_pos, joint_vel)

        # Apply tanh-based smoothing: tau_out = tanh(w * tau_in) / w
        # self.smoothing_coefficients has shape (num_joints,), broadcast to (num_envs, num_joints)
        w = self.smoothing_coefficients.unsqueeze(0)  # shape: (1, num_joints)
        tau_in = control_action_sim.joint_efforts  # shape: (num_envs, num_joints)

        # Compute smoothed torques
        smoothed_torques = torch.tanh(w * tau_in) / w

        # Update control action with smoothed torques
        control_action_sim.joint_efforts = smoothed_torques

        return control_action_sim
