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
parser.add_argument("--input_data", type=str, default="5Hz_sine_data.pt",
                    help="Input trajectory data file (in data/DM8006/)")
parser.add_argument("--output_suffix", type=str, default="sim_output",
                    help="Output file suffix (will create <input_name>_<suffix>.pt)")
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

    armature = torch.tensor([0.0055] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    damping = torch.tensor([2.7120e-06] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    friction = torch.tensor([0.4504] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    bias = torch.tensor([-0.0467] * 1, device=env.unwrapped.device).unsqueeze(0)
    time_lag = torch.tensor([[3.486177682876587]], dtype=torch.int, device=env.unwrapped.device)
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

    data_dir = project_root() / "data" / "DM8006"

    # Load trajectory from real data
    input_data_path = data_dir / args_cli.input_data
    if not input_data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {input_data_path}")

    input_data = torch.load(input_data_path)

    # 从真实数据中获取输入轨迹
    time_data_real = input_data["time"]
    des_dof_pos_real = input_data["des_dof_pos"]  # (num_steps, 1)

    # 转换到仿真设备
    t = time_data_real.to(env.unwrapped.device)
    num_steps = len(t)
    sample_rate = 1 / (t[1] - t[0])  # Hz

    # 使用真实数据的输入作为轨迹
    trajectory = des_dof_pos_real.to(env.unwrapped.device)

    # 如果有多个关节，复制到所有关节
    if len(joint_ids) > 1:
        trajectory = trajectory.repeat(1, len(joint_ids))

    print(f"[INFO]: 加载真实数据轨迹: {args_cli.input_data}")
    print(f"[INFO]: 时间范围: {t[0]:.4f} - {t[-1]:.4f} 秒")
    print(f"[INFO]: 采样率: {sample_rate:.1f} Hz")
    print(f"[INFO]: 数据点数: {num_steps}")
    # init position - 使用轨迹的第一个点初始化
    if trajectory.shape[1] == len(joint_ids):
        init_pos = trajectory[0, :]
    else:
        init_pos = trajectory[0, 0].repeat(len(joint_ids))

    articulation.write_joint_position_to_sim(init_pos.unsqueeze(0), joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim(torch.zeros((1, len(joint_ids)), device=env.unwrapped.device))
    articulation.write_data_to_sim()

    counter = 0
    # simulate environment
    dof_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    dof_target_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            dof_pos_buffer[counter, :] = env.unwrapped.scene.articulations["robot"].data.joint_pos[0, joint_ids] - bias[0]
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

            # 获取当前时刻的轨迹点
            if trajectory.shape[1] == len(joint_ids):
                current_traj = trajectory[counter % num_steps, :]
            else:
                # 单关节轨迹，复制到所有关节
                current_traj = trajectory[counter % num_steps, 0].repeat(len(joint_ids))

            actions = current_traj.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # apply actions
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

    # 生成输出文件名
    input_name = args_cli.input_data.replace('.pt', '')
    output_file = data_dir / f"{input_name}_{args_cli.output_suffix}.pt"

    torch.save({
        "time": t.cpu(),
        "dof_pos": dof_pos_buffer.cpu(),
        "des_dof_pos": dof_target_pos_buffer.cpu(),
    }, output_file)

    print(f"[INFO]: 仿真数据已保存到: {output_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
