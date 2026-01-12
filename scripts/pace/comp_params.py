# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to compare fitted parameters with real trajectory using 5Hz sine input."""

"""Launch Isaac Sim Simulator first."""

import argparse
import re
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Compare fitted parameters with real trajectory.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Y1-1-v0", help="Name of the task.")
parser.add_argument("--folder_name", type=str, default=None, help="Name of the folder to use.")
parser.add_argument("--mean_name", type=str, default=None, help="Name of the parameters file to use.")
parser.add_argument("--robot_name", type=str, default="Y1_1_sim", help="Name of the robot.")
parser.add_argument("--frequency", type=float, default=5.0, help="Sine wave frequency in Hz.")
parser.add_argument("--amplitude", type=float, default=0.5, help="Sine wave amplitude in radians.")
parser.add_argument("--duration", type=float, default=10.0, help="Duration of the test in seconds.")
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
import matplotlib.pyplot as plt
from torch import pi

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Y1_1Pace.tasks  # noqa: F401

_pattern = re.compile(r"^mean_(\d+)\.pt$")


def find_latest_params(root: Path):
    best = None  # tuple (int, Path)
    for p in root.rglob("mean_*.pt"):
        m = _pattern.match(p.name)
        if not m:
            continue
        num = int(m.group(1))
        if best is None or num > best[0]:
            best = (num, p)
    if best is None:
        return None, None
    return best[1], best[0]


def main():
    # Load fitted parameters
    current_dir = Path(__file__).parent.resolve()
    project_root_dir = current_dir.parent.parent
    log_dir = project_root_dir / "logs" / "pace" / args_cli.robot_name

    if not log_dir.exists():
        raise FileNotFoundError(f"No logs for robot {args_cli.robot_name} under {log_dir}")

    # if no folder_name given, pick the most recent run folder for the robot
    folder_name = args_cli.folder_name
    if not folder_name:
        robot_dir = log_dir
        candidates = [p for p in robot_dir.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No run folders found under {robot_dir}")
        latest_folder = max(candidates, key=lambda p: p.stat().st_mtime)
        folder_name = latest_folder.name

    # now point log_dir at the chosen robot/run folder
    log_dir = log_dir / folder_name

    if args_cli.mean_name is None:
        params_path, _ = find_latest_params(log_dir)
    else:
        params_path = log_dir / args_cli.mean_name
        if not params_path.exists():
            raise FileNotFoundError(f"Given params file {params_path} does not exist")
    if params_path is None:
        raise FileNotFoundError(f"No mean_*.pt files found under {log_dir}")
    print(f"[INFO] Loading parameters from: {params_path}")
    mean = torch.load(params_path)
    config = torch.load(log_dir / "config.pt")
    joint_order = config["joint_order"]

    # Extract parameters
    num_joints = len(joint_order)
    armature_params = mean[:num_joints]
    damping_params = mean[num_joints:2 * num_joints]
    friction_params = mean[2 * num_joints:3 * num_joints]
    bias_params = mean[3 * num_joints:4 * num_joints]
    delay_param = mean[-1].item()

    print(f"[INFO] Best parameter set: {mean}")
    print(f"[INFO] Armature params: {armature_params}")
    print(f"[INFO] Viscous friction params: {damping_params}")
    print(f"[INFO] Static friction params: {friction_params}")
    print(f"[INFO] Encoder bias params: {bias_params}")
    print(f"[INFO] Delay param: {delay_param}")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    articulation = env.unwrapped.scene["robot"]
    joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)

    # Reset environment
    env.reset()

    # Apply fitted parameters to simulation
    print("[INFO] Applying fitted parameters to simulation...")
    articulation.write_joint_armature_to_sim(armature_params.unsqueeze(0), joint_ids=joint_ids, env_ids=torch.arange(args_cli.num_envs, device=env.unwrapped.device))
    articulation.data.default_joint_armature[:, joint_ids] = armature_params.unsqueeze(0)

    articulation.write_joint_viscous_friction_coefficient_to_sim(damping_params.unsqueeze(0), joint_ids=joint_ids, env_ids=torch.arange(args_cli.num_envs, device=env.unwrapped.device))
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping_params.unsqueeze(0)

    articulation.write_joint_friction_coefficient_to_sim(friction_params.unsqueeze(0), joint_ids=joint_ids, env_ids=torch.arange(args_cli.num_envs, device=env.unwrapped.device))
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction_params.unsqueeze(0)

    # Update actuator parameters
    drive_types = articulation.actuators.keys()
    time_lag = torch.tensor([[int(delay_param)]], dtype=torch.int, device=env.unwrapped.device)

    for drive_type in drive_types:
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_time_lags(time_lag)
        articulation.actuators[drive_type].update_encoder_bias(bias_params[drive_joint_idx].unsqueeze(0))
        articulation.actuators[drive_type].reset(torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device))

    # Generate 5Hz sine wave trajectory
    duration = args_cli.duration
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=env.unwrapped.device)

    frequency = args_cli.frequency
    amplitude = args_cli.amplitude

    # Sine wave: amplitude * sin(2*pi*frequency*t)
    sine_signal = amplitude * torch.sin(2 * pi * frequency * t)

    # Create trajectory for all joints
    trajectory = torch.zeros((num_steps, len(joint_ids)), device=env.unwrapped.device)

    # Get initial joint positions
    init_pos = articulation.data.joint_pos[0, joint_ids].clone()

    # Apply sine wave to trajectory
    for i in range(len(joint_ids)):
        trajectory[:, i] = init_pos[i] + sine_signal

    # Initialize data recording
    sim_positions = []
    sim_velocities = []
    target_positions = []
    time_stamps = []

    print(f"[INFO] Running simulation with {frequency}Hz sine wave for {duration} seconds...")

    # Run simulation
    for step in range(num_steps):
        # Get target position for this step
        target_pos = trajectory[step, :].unsqueeze(0)  # shape: (1, num_joints)

        # Create action (position command)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        actions[0, :len(joint_ids)] = target_pos[0, :]

        # Step simulation
        env.step(actions)

        # Record data
        current_pos = articulation.data.joint_pos[0, joint_ids].clone()
        current_vel = articulation.data.joint_vel[0, joint_ids].clone()

        sim_positions.append(current_pos.cpu())
        sim_velocities.append(current_vel.cpu())
        target_positions.append(target_pos[0, :].cpu())
        time_stamps.append(t[step].cpu().item())

        # Print progress
        if (step + 1) % 1000 == 0:
            print(f"[INFO] Progress: {step + 1}/{num_steps} steps")

    # Convert to tensors
    sim_positions = torch.stack(sim_positions)  # (num_steps, num_joints)
    sim_velocities = torch.stack(sim_velocities)
    target_positions = torch.stack(target_positions)
    time_stamps = torch.tensor(time_stamps)

    print("[INFO] Simulation complete. Plotting results...")

    # Plot results
    for i in range(len(joint_ids)):
        # Position plot
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(time_stamps.numpy(), target_positions[:, i].numpy(), 'k--', label='Target', linewidth=2, alpha=0.7)
        plt.plot(time_stamps.numpy(), sim_positions[:, i].numpy(), 'b-', label='Simulation', linewidth=1.5)
        plt.title(f"Joint {joint_order[i]} - Position Tracking ({frequency}Hz Sine Wave)")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Velocity plot
        plt.subplot(2, 1, 2)
        plt.plot(time_stamps.numpy(), sim_velocities[:, i].numpy(), 'r-', label='Velocity', linewidth=1.5)
        plt.title(f"Joint {joint_order[i]} - Velocity")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [rad/s]")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Calculate tracking error
    position_error = sim_positions - target_positions
    rmse = torch.sqrt(torch.mean(position_error ** 2, dim=0))

    print("\n[INFO] Tracking Performance:")
    for i, joint_name in enumerate(joint_order):
        print(f"  {joint_name}: RMSE = {rmse[i].item():.6f} rad")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
