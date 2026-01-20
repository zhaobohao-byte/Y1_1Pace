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
parser.add_argument("--input_data", type=str, default="/home/bohao/LuvRobot/Y1_1Pace/data/Atom3motors/raw_pt/260117_chrip_20s_3motors_aligned.pt",
                    help="Input trajectory data file path (absolute or relative to data/). If not provided, interactive mode will start.")
parser.add_argument("--params_file", type=str, default="/home/bohao/LuvRobot/Y1_1Pace/logs/pace/Atom3DOF_sim/26_01_19_12-17-39/mean_299.pt",
                    help="Path to optimized parameters file (e.g., mean_299.pt). If not provided, will use default parameters.")
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


def select_from_menu(options, title="Please select:"):
    """Interactive menu selection with fallback."""
    try:
        import inquirer
        answers = inquirer.prompt([inquirer.List('selection', message=title, choices=options)])
        return answers['selection'] if answers else None
    except ImportError:
        print(f"\n{title}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        try:
            choice = input("\nEnter number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            return options[idx] if 0 <= idx < len(options) else None
        except (ValueError, KeyboardInterrupt):
            return None


def select_data_file_interactive(data_dir):
    """Interactive selection of data file from folders."""
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"No subdirectories found in {data_dir}")
        return None

    print("\n" + "=" * 70 + "\nSelect Data Folder\n" + "=" * 70)
    selected_folder = select_from_menu([d.name for d in subdirs], "Select folder:")
    if not selected_folder:
        return None

    folder_path = data_dir / selected_folder
    pt_files = sorted([f.name for f in folder_path.glob("*.pt")])
    if not pt_files:
        print(f"No .pt files found in {folder_path}")
        return None

    print(f"\nFolder: {selected_folder}")
    selected_file = select_from_menu(pt_files, "Select data file:")
    return folder_path / selected_file if selected_file else None


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

    # Load parameters from file if provided, otherwise use defaults
    from pathlib import Path
    params_path = Path(args_cli.params_file)
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    
    print(f"[INFO]: Loading parameters from: {params_path}")
    mean_params = torch.load(params_path, map_location=env.unwrapped.device)

    # Parse parameters according to PACE format:
    # [armature (n), damping (n), friction (n), bias (n), delay (1)]
    n_joints = len(joint_ids)
    armature = mean_params[:n_joints].unsqueeze(0)
    damping = mean_params[n_joints:2 * n_joints].unsqueeze(0)
    friction = mean_params[2 * n_joints:3 * n_joints].unsqueeze(0)
    bias = mean_params[3 * n_joints:4 * n_joints].unsqueeze(0)
    time_lag = mean_params[-1:].unsqueeze(0).to(torch.int)

    print("[INFO]: Loaded parameters:")
    print(f"  Armature: {armature.squeeze().tolist()}")
    print(f"  Damping: {damping.squeeze().tolist()}")
    print(f"  Friction: {friction.squeeze().tolist()}")
    print(f"  Bias: {bias.squeeze().tolist()}")
    print(f"  Time lag: {time_lag.item()}")    
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

    data_dir = project_root() / "data" 

    # Load trajectory from real data
    if args_cli.input_data is None:
        # Interactive mode: select file from folders
        print("\n" + "=" * 70)
        print("Interactive Data Collection Mode")
        print("=" * 70)

        input_data_path = select_data_file_interactive(data_dir)
        if input_data_path is None:
            print("No file selected. Exiting.")
            env.close()
            return

        print(f"\n[INFO]: Selected input file: {input_data_path.relative_to(data_dir)}")

        # Ask for output suffix in interactive mode
        custom_suffix = input("\nOutput suffix (press Enter for default 'sim_output'): ").strip()
        output_suffix = custom_suffix if custom_suffix else "sim_output"
        print(f"[INFO]: Output suffix: {output_suffix}")
    else:
        # Command line mode: use provided path
        from pathlib import Path
        input_path = Path(args_cli.input_data)
        
        if input_path.is_absolute():
            # Absolute path provided
            input_data_path = input_path
        elif "/" in args_cli.input_data or "\\" in args_cli.input_data:
            # Relative path with folder structure
            input_data_path = data_dir / args_cli.input_data
        else:
            # No folder specified, default to raw_data folder
            input_data_path = data_dir / "raw_data" / args_cli.input_data

        if not input_data_path.exists():
            raise FileNotFoundError(f"Input data file not found: {input_data_path}")

        output_suffix = args_cli.output_suffix
        print(f"[INFO]: Using input file: {input_data_path}")

    input_data = torch.load(input_data_path)

    # 从真实数据中获取输入轨迹
    device = env.unwrapped.device
    t = input_data["time"].to(device)
    des_dof_pos_real = input_data["des_dof_pos"]  # (num_steps, n_joints)
    dof_pos_real = input_data["dof_pos"]  # (num_steps, n_joints)

    # 检查是否有速度数据
    has_velocity = "dof_vel" in input_data
    if has_velocity:
        dof_vel_real = input_data["dof_vel"]  # (num_steps, n_joints)
        print("[INFO]: 发现速度数据，将进行位置和速度比较")
    else:
        dof_vel_real = None
        print("[INFO]: 未发现速度数据，仅进行位置比较")

    num_steps = len(t)
    sample_rate = 1 / (t[1] - t[0])  # Hz
    n_data_joints = des_dof_pos_real.shape[1]

    # 处理轨迹数据：支持单关节或多关节
    if n_data_joints == len(joint_ids):
        # 数据关节数匹配
        trajectory = des_dof_pos_real.to(device)
        init_pos = dof_pos_real[0, :].to(device)
        init_vel = dof_vel_real[0, :].to(device) if has_velocity else torch.zeros(len(joint_ids), device=device)
    elif n_data_joints == 1:
        # 单关节数据，复制到所有关节
        trajectory = des_dof_pos_real.repeat(1, len(joint_ids)).to(device)
        init_pos = dof_pos_real[0, 0].repeat(len(joint_ids)).to(device)
        init_vel = dof_vel_real[0, 0].repeat(len(joint_ids)).to(device) if has_velocity else torch.zeros(len(joint_ids), device=device)
    else:
        raise ValueError(f"数据关节数 ({n_data_joints}) 与配置关节数 ({len(joint_ids)}) 不匹配")

    print("[INFO]: 加载真实数据轨迹")
    print(f"[INFO]: 时间范围: {t[0].item():.4f} - {t[-1].item():.4f} 秒")
    print(f"[INFO]: 采样率: {sample_rate.item():.1f} Hz")
    print(f"[INFO]: 数据点数: {num_steps}")
    print(f"[INFO]: 数据关节数: {n_data_joints}, 仿真关节数: {len(joint_ids)}")

    print(f"[INFO]: 初始位置（实际测量值）: {init_pos.cpu().numpy()}")
    if has_velocity:
        print(f"[INFO]: 初始速度（实际测量值）: {init_vel.cpu().numpy()}")
    
    # 设置初始位置时需要加上编码器偏置（因为仿真器内部会减去偏置）
    # 实际位置 = 仿真位置 - bias，所以仿真位置 = 实际位置 + bias
    init_pos_with_bias = init_pos + bias.squeeze()
    print(f"[INFO]: 初始位置（加上偏置后）: {init_pos_with_bias.cpu().numpy()}")
    
    articulation.write_joint_position_to_sim(init_pos_with_bias.unsqueeze(0), joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim(init_vel.unsqueeze(0), joint_ids=joint_ids)
    articulation.write_data_to_sim()

    # 初始化缓冲区
    dof_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=device)
    dof_vel_buffer = torch.zeros(num_steps, len(joint_ids), device=device)
    dof_target_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=device)

    counter = 0
    while simulation_app.is_running() and counter < num_steps:
        with torch.inference_mode():
            robot = env.unwrapped.scene.articulations["robot"]
            
            # 记录当前状态（处理多关节bias）
            dof_pos_buffer[counter] = robot.data.joint_pos[0, joint_ids] - bias[0, :len(joint_ids)]
            dof_vel_buffer[counter] = robot.data.joint_vel[0, joint_ids]
            
            # 设置目标动作
            actions = trajectory[counter].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            
            # 执行步进
            env.step(actions)
            dof_target_pos_buffer[counter] = robot._data.joint_pos_target[0, joint_ids]
            
            counter += 1
            if counter % 500 == 0:
                print(f"[INFO]: Step {counter/sample_rate.item():.2f} seconds")

    env.close()

    # 保存仿真数据
    output_dir = data_dir / "sim_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_data_path.stem}_{output_suffix}.pt"

    save_data = {
        "time": t.cpu(),
        "dof_pos": dof_pos_buffer.cpu(),
        "dof_vel": dof_vel_buffer.cpu(),
        "des_dof_pos": dof_target_pos_buffer.cpu(),
    }
    torch.save(save_data, output_file)

    print(f"[INFO]: 仿真数据已保存到: {output_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
