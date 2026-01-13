# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Y1-1-v0", help="Name of the task.")
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum frequency for the chirp signal in Hz.")
parser.add_argument("--max_frequency", type=float, default=10.0, help="Maximum frequency for the chirp signal in Hz.")
parser.add_argument("--duration", type=float, default=20.0, help="Duration of the chirp signal in seconds.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from torch import pi

import Y1_1Pace.tasks  # noqa: F401
from Y1_1Pace.utils import project_root


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    articulation = env.unwrapped.scene["robot"]

    joint_order = env_cfg.sim2real.joint_order
    joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)

    armature = torch.tensor([0.1] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    damping = torch.tensor([4.5] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    friction = torch.tensor([0.05] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    bias = torch.tensor([0.05] * 12, device=env.unwrapped.device).unsqueeze(0)
    time_lag = torch.tensor([[5]], dtype=torch.int, device=env.unwrapped.device)
    env.reset()

    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature)))
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping)))
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0]))
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction
    drive_types = articulation.actuators.keys()
    for drive_type in drive_types:
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_time_lags(time_lag)
        articulation.actuators[drive_type].update_encoder_bias(bias[:, drive_joint_idx])
        articulation.actuators[drive_type].reset(torch.arange(env.unwrapped.num_envs))

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name

    # Create a chirp signal for each action dimension

    duration = args_cli.duration  # seconds
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()  # Hz
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=env.unwrapped.device)
    f0 = args_cli.min_frequency  # Hz
    f1 = args_cli.max_frequency  # Hz

    # Linear chirp: phase = 2*pi*(f0*t + (f1-f0)/(2*duration)*t^2)
    phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t ** 2)
    chirp_signal = torch.sin(phase)

    trajectory = torch.zeros((num_steps, len(joint_ids)), device=env.unwrapped.device)
    trajectory[:, :] = chirp_signal.unsqueeze(-1)

    phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t ** 2)
    chirp_signal = torch.sin(phase)

    trajectory = torch.zeros((num_steps, len(joint_ids)), device=env.unwrapped.device)
    trajectory[:, :] = chirp_signal.unsqueeze(-1)
    trajectory_directions = torch.tensor(
        [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        device=env.unwrapped.device
    )
    trajectory_bias = torch.tensor(
        [0.0, 0.4, 0.8] * 4,
        device=env.unwrapped.device
    )
    trajectory_scale = torch.tensor(
        [0.25, 0.5, -2.0] * 4,
        device=env.unwrapped.device
    )
    trajectory[:, joint_ids] = (trajectory[:, joint_ids] + trajectory_bias.unsqueeze(0)) * trajectory_directions.unsqueeze(0) * trajectory_scale.unsqueeze(0)

    articulation.write_joint_position_to_sim(trajectory[0, :].unsqueeze(0) + bias[0, joint_ids])
    articulation.write_joint_velocity_to_sim(torch.zeros((1, len(joint_ids)), device=env.unwrapped.device))
    
    counter = 0
    # simulate environment
    dof_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    dof_vel_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    dof_target_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    time_data = t

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Record joint states (position and velocity)
            dof_pos_buffer[counter, :] = env.unwrapped.scene.articulations["robot"].data.joint_pos[0, joint_ids] - bias[0]
            dof_vel_buffer[counter, :] = env.unwrapped.scene.articulations["robot"].data.joint_vel[0, joint_ids]

            # Compute actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions = trajectory[counter % num_steps, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

            # Apply actions
            obs, _, _, _, _ = env.step(actions)
            dof_target_pos_buffer[counter, :] = env.unwrapped.scene.articulations["robot"]._data.joint_pos_target[0, joint_ids]

            counter += 1
            if counter % 500 == 0:
                print(f"[INFO]: Step {counter/sample_rate:.2f} seconds")
            if counter >= num_steps:
                break

    # close the simulator
    env.close()

    from time import sleep
    sleep(1)  # wait a bit for everything to settle

    (data_dir).mkdir(parents=True, exist_ok=True)

    # Save data with velocity information
    data_dict = {
        "time": time_data.cpu(),
        "dof_pos": dof_pos_buffer.cpu(),
        "dof_vel": dof_vel_buffer.cpu(),
        "des_dof_pos": dof_target_pos_buffer.cpu(),
    }

    save_path = data_dir / "chrip_sin_traj_data.pt"
    torch.save(data_dict, save_path)
    print(f"[INFO]: Data saved to {save_path}")
    print("[INFO]: Data contains: time, dof_pos, dof_vel, des_dof_pos")
    print(f"[INFO]: Total samples: {num_steps}, Duration: {duration:.2f}s, Sample rate: {sample_rate:.2f}Hz")

    import matplotlib.pyplot as plt

    # Plot position and velocity for each joint
    for i in range(len(joint_ids)):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Position plot
        ax1.plot(t.cpu().numpy(), dof_pos_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} pos", linewidth=2)
        ax1.plot(t.cpu().numpy(), dof_target_pos_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} target", linestyle='--', linewidth=1.5)
        ax1.set_title(f"Joint {joint_order[i]} - Position", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Position [rad]")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Velocity plot
        ax2.plot(t.cpu().numpy(), dof_vel_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} vel", linewidth=2, color='orange')
        ax2.set_title(f"Joint {joint_order[i]} - Velocity", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Velocity [rad/s]")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
