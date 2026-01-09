# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Y1-1-v0", help="Name of the task.")
parser.add_argument("--constant_velocity", type=float, default=15, help="Constant velocity for joint motion (rad/s). If None, uses zero actions.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# ensure headless is False (close headless mode)
args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Y1_1Pace.tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, _ = env.reset()

    # constant velocity motion setup with torque control
    target_velocity = None
    current_velocity = 0.0
    if args_cli.constant_velocity is not None:
        target_velocity = args_cli.constant_velocity
        print(f"[INFO]: Starting velocity ramp to {target_velocity} rad/s with torque control")

    # get simulation timestep
    sim_dt = env.unwrapped.cfg.sim.dt

    # data recording for velocity-time and torque-time curves
    velocity_data = []
    torque_data = []
    time_data = []
    recording_duration = 6.0
    max_steps = int(recording_duration / sim_dt)

    # velocity ramp parameters
    ramp_duration = 2.50
    ramp_steps = int(ramp_duration / sim_dt)
    velocity_increment = None
    if target_velocity is not None:
        velocity_increment = target_velocity / ramp_steps

    # PD controller parameters for velocity control
    kp = 20.0  # proportional gain

    # get robot articulation
    robot = env.unwrapped.scene["robot"]
    joint_indices = torch.arange(robot.num_joints, device=env.unwrapped.device)

    # simulate environment
    step_count = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if target_velocity is not None:
                # velocity ramp: linearly increase velocity to target
                if step_count < ramp_steps:
                    # ramp phase: increase velocity
                    current_velocity = velocity_increment * step_count
                else:
                    # constant velocity phase: maintain target velocity
                    current_velocity = target_velocity

                # get current joint velocity - shape: (num_envs, num_joints)
                current_vel = robot.data.joint_vel[:, joint_indices].clone()
                # compute velocity error
                velocity_error = current_velocity - current_vel
                # P controller: compute torque to achieve target velocity
                # Torque = Kp * (v_target - v_current)
                torque = kp * velocity_error
                actions = torque
            else:
                # compute zero actions (zero torque)
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            obs, _, _, _, _ = env.step(actions)

            # record velocity and torque data (only for 10 seconds)
            if step_count < max_steps:
                joint_vel = robot.data.joint_vel[0, joint_indices].cpu().numpy()  # [num_joints]
                # get actual applied joint torques from physics simulation
                # In torque control mode, get the actual torques applied (may be different from commands due to saturation)
                try:
                    # Try to get actual applied torques from root_physx_view using get_dof_actuation_force
                    actual_joint_torques = robot.root_physx_view.get_dof_actuation_force()[0, joint_indices].cpu().numpy()
                except (AttributeError, TypeError):
                    # Fallback: try to get from robot.data if available
                    try:
                        # Some versions may have applied_torque in data
                        actual_joint_torques = robot.data.applied_torque[0, joint_indices].cpu().numpy()
                    except AttributeError:
                        # Final fallback: use actions (commanded torques)
                        # Note: In torque control without saturation, actual = commanded
                        actual_joint_torques = actions[0, joint_indices].cpu().numpy()
                current_time = step_count * sim_dt

                # record data for each joint
                for i in range(len(joint_indices)):
                    velocity_data.append(joint_vel[i])
                    # Use actual applied torque
                    torque_val = actual_joint_torques[i] if i < len(actual_joint_torques) else 0.0
                    torque_data.append(torque_val)
                    time_data.append(current_time)

            step_count += 1

            # stop recording after 10 seconds
            if step_count >= max_steps:
                print(f"[INFO]: Recorded data for {recording_duration} seconds ({step_count} steps)")
                break

    # close the simulator
    env.close()

    # plot velocity-time and torque-time curves on the same time axis
    if len(velocity_data) > 0:
        print(f"[INFO]: Recorded {len(velocity_data)} data points")
        velocity_array = np.array(velocity_data)
        torque_array = np.array(torque_data)
        time_array = np.array(time_data)

        # create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # plot velocity on left y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel("Time [s]", fontsize=12)
        ax1.set_ylabel("Joint Velocity [rad/s]", color=color1, fontsize=12)
        line1 = ax1.plot(time_array, velocity_array, color=color1, linewidth=1.5, label='Velocity')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        # plot torque on right y-axis
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel("Joint Torque [Nm]", color=color2, fontsize=12)
        line2 = ax2.plot(time_array, torque_array, color=color2, linewidth=1.5, label='Torque')
        ax2.tick_params(axis='y', labelcolor=color2)

        # add title and legend
        plt.title("Velocity and Torque vs Time", fontsize=14, pad=20)

        # combine legends
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.show()

        # plot torque-velocity scatter plot
        fig2, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(velocity_array, torque_array, alpha=0.5, s=1, c='purple')
        ax3.set_xlabel("Joint Velocity [rad/s]", fontsize=12)
        ax3.set_ylabel("Joint Torque [Nm]", fontsize=12)
        ax3.set_title("Torque-Velocity Scatter Plot", fontsize=14)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"[INFO]: Velocity range: [{velocity_array.min():.4f}, {velocity_array.max():.4f}] rad/s")
        print(f"[INFO]: Torque range: [{torque_array.min():.4f}, {torque_array.max():.4f}] Nm")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
