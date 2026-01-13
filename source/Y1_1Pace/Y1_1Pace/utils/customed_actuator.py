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


class PaceDCMotor(DCMotor):
    """Pace DC Motor actuator model with first-order low-pass filter.

    The actuator models a DC motor with a first-order low-pass filter applied to the
    torque output. This simulates the dynamic response characteristics of real actuators,
    providing smoother torque transitions and more realistic behavior.

    The filter is implemented using a discrete-time first-order IIR filter:
        y[n] = alpha * x[n] + (1 - alpha) * y[n-1]

    where alpha = dt / (dt + tau), tau = 1 / (2 * pi * cutoff_frequency)
    """

    cfg: CustomedDCMotorCfg

    def __init__(self, cfg: CustomedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # Calculate filter coefficient alpha
        # tau = 1 / (2 * pi * cutoff_frequency)
        # alpha = dt / (dt + tau)
        dt = self._sim_params.dt  # simulation time step
        tau = 1.0 / (2.0 * torch.pi * cfg.cutoff_frequency)
        self.alpha = dt / (dt + tau)

        # Initialize previous torque values (for filter state)
        self.prev_torques = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
            dtype=torch.float32
        )

        print(f"[PaceDCMotor] Initialized with cutoff frequency: {cfg.cutoff_frequency} Hz")
        print(f"[PaceDCMotor] Filter coefficient alpha: {self.alpha:.6f}")
        print(f"[PaceDCMotor] Time constant tau: {tau:.6f} s")

    def reset(self, env_ids: Sequence[int]):
        """Reset the actuator state for specified environments."""
        super().reset(env_ids)
        # Reset filter state for specified environments
        self.prev_torques[env_ids] = 0.0

    def update_cutoff_frequency(self, cutoff_frequency: float):
        """Update the cutoff frequency of the low-pass filter.

        Args:
            cutoff_frequency: New cutoff frequency in Hz
        """
        dt = self._sim_params.dt
        tau = 1.0 / (2.0 * torch.pi * cutoff_frequency)
        self.alpha = dt / (dt + tau)
        print(f"[PaceDCMotor] Updated cutoff frequency to: {cutoff_frequency} Hz (alpha: {self.alpha:.6f})")

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Compute the actuator torques with low-pass filtering.

        Args:
            control_action: The control action containing desired joint positions
            joint_pos: Current joint positions
            joint_vel: Current joint velocities

        Returns:
            Filtered control action with smoothed torques
        """
        # Compute actuator model (DC motor with PD control)
        control_action_sim = super().compute(control_action, joint_pos, joint_vel)

        # Apply first-order low-pass filter to torques
        # y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        filtered_torques = (self.alpha * control_action_sim.joint_efforts
                            + (1.0 - self.alpha) * self.prev_torques)

        # Update previous torques for next iteration
        self.prev_torques = filtered_torques.clone()

        # Update control action with filtered torques
        control_action_sim.joint_efforts = filtered_torques

        return control_action_sim
