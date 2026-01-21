# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Y1-1-v0", help="Name of the task.")
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


from isaaclab_tasks.utils import parse_env_cfg

import Y1_1Pace.tasks  # noqa: F401
from Y1_1Pace.utils import project_root
from Y1_1Pace import CMAESOptimizer

import isaaclab_tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Create optimization
    bounds_params = env_cfg.sim2real.bounds_params.to(env.unwrapped.device)
    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    sim_joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)

    data_file = project_root() / "data" / env_cfg.sim2real.data_dir
    log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

    data = torch.load(data_file)
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)

    # 初始位置使用实际测量值（update_simulator会自动加上优化中的bias参数）
    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

    time_steps = time_data.shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt

    # Get optimization weights from config if available, otherwise use defaults
    pos_weight = getattr(env_cfg.sim2real.cmaes, 'pos_weight', 1.0)
    vel_weight = getattr(env_cfg.sim2real.cmaes, 'vel_weight', 0.02)    # 'vel_weight', 0.1
    smoothness_weight = getattr(env_cfg.sim2real.cmaes, 'smoothness_weight', 0.1)  # Smoothness penalty weight (default 0.01)

    # Determine if velocity should be used: both data available AND vel_weight is set
    has_velocity_data = "dof_vel" in data
    use_velocity = has_velocity_data and vel_weight is not None

    if use_velocity:
        measured_dof_vel = data["dof_vel"].to(env.unwrapped.device)
        print(f"[INFO]: Using velocity in optimization. Position weight: {pos_weight}, Velocity weight: {vel_weight}, Smoothness weight: {smoothness_weight}")
    else:
        measured_dof_vel = None
        if not has_velocity_data:
            print("[INFO]: No velocity data found in file. Optimization will use position error only.")
        elif vel_weight is None or vel_weight == 0.0:
            print(f"[INFO]: Velocity weight is {vel_weight}. Optimization will use position error only.")
        print(f"[INFO]: Optimization weights - Position: {pos_weight}, Smoothness: {smoothness_weight}")

    opt = CMAESOptimizer(
        bounds=bounds_params,
        population_size=env.unwrapped.num_envs,
        log_dir=log_dir,
        joint_order=joint_order,
        max_iteration=env_cfg.sim2real.cmaes.max_iteration,
        data=data,
        device=env.unwrapped.device,
        epsilon=env_cfg.sim2real.cmaes.epsilon,
        sigma=env_cfg.sim2real.cmaes.sigma,
        save_interval=env_cfg.sim2real.cmaes.save_interval,
        save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
        pos_weight=pos_weight,
        vel_weight=vel_weight if use_velocity else None,
        smoothness_weight=smoothness_weight,
    )

    env.reset()
    opt.update_simulator(articulation, sim_joint_ids, initial_dof_pos)

    counter = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Get current simulated joint states
            sim_joint_pos = env.unwrapped.scene.articulations["robot"].data.joint_pos[:, sim_joint_ids]
            sim_joint_vel = env.unwrapped.scene.articulations["robot"].data.joint_vel[:, sim_joint_ids]

            # Get measured/target states for current timestep
            measured_pos = measured_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

            # Update optimizer with position and optionally velocity data
            if use_velocity:
                measured_vel = measured_dof_vel[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
                opt.tell(sim_joint_pos, measured_pos, sim_joint_vel, measured_vel)
            else:
                opt.tell(sim_joint_pos, measured_pos, sim_joint_vel, None)

            # Set target position actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, sim_joint_ids] = target_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

            # apply actions
            env.step(actions)
            counter += 1

            if counter % 500 == 0:
                print(f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds ({counter / time_steps * 100:.1f} %)")

            if counter >= time_steps:
                print("[INFO]: Reached the end of the trajectory.")
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                env.reset()
                opt.update_simulator(env.unwrapped.scene["robot"], sim_joint_ids, initial_dof_pos)
    # close optimizer
    opt.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
