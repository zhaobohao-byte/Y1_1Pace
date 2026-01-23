#!/usr/bin/env python3

import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


# 配置 - 根据需要修改这部分就好
# 示例关节配置：
# JOINT_ORDER = ['right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint']  # 三关节
JOINT_ORDER = ['left_hip_roll_joint']  # 单关节

# 使用脚本所在位置作为基准，这样无论从哪里运行都能找到文件
SCRIPT_DIR = Path(__file__).parent
RAW_CSV_DIR = SCRIPT_DIR / "../../data/RS_motors/raw_csv"
OUTPUT_DIR = SCRIPT_DIR / "../../data/RS_motors/raw_pt"


def convert_csv_to_pt(csv_path):
    """把单个 csv 转成 .pt 文件"""
    print(f"\n处理: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path)

        time = torch.tensor(df['time'].values, dtype=torch.float32)
        n = len(df)

        # 创建张量，按指定关节顺序排列
        n_joints = len(JOINT_ORDER)
        des_pos = torch.zeros((n, n_joints), dtype=torch.float32)
        act_pos = torch.zeros((n, n_joints), dtype=torch.float32)
        vel = torch.zeros((n, n_joints), dtype=torch.float32)

        for i, joint in enumerate(JOINT_ORDER):
            if f'des_dof_pos_{joint}' in df.columns:
                des_pos[:, i] = torch.tensor(df[f'des_dof_pos_{joint}'].values)
            if f'dof_pos_{joint}' in df.columns:
                act_pos[:, i] = torch.tensor(df[f'dof_pos_{joint}'].values)
            if f'dof_vel_{joint}' in df.columns:
                vel[:, i] = torch.tensor(df[f'dof_vel_{joint}'].values)

        # 保存
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = OUTPUT_DIR / f"{csv_path.stem}.pt"

        # 创建输出数据字典
        data_dict = {'time': time, 'des_dof_pos': des_pos, 'dof_pos': act_pos}

        # 只有当速度数据有效时才保存
        if vel.norm() > 0:  # 检查是否有速度数据
            data_dict['dof_vel'] = vel

        torch.save(data_dict, save_path)

        print(f"  → 已保存: {save_path.name}  ({len(time)} 帧)")
        return save_path

    except Exception as e:
        print(f"  × 转换失败: {e}")
        return None


def plot_trajectory(pt_path):
    """画轨迹图：每个关节一行，左边位置对比，右边速度"""
    data = torch.load(pt_path)
    t = data['time'].numpy()

    # 自动根据关节数量调整布局
    n_joints = len(JOINT_ORDER)
    # 2列（位置和速度），关节数量行
    fig, axes = plt.subplots(n_joints, 2, figsize=(14, 4.5 * n_joints))
    fig.suptitle(f'Joint Tracking - {pt_path.stem}', fontsize=14, fontweight='bold')

    # 确保axes是二维数组，即使只有一个关节
    if n_joints == 1:
        axes = axes.reshape(1, -1)

    # 检查是否有速度数据
    has_velocity = 'dof_vel' in data and data['dof_vel'].norm() > 0

    for i, name in enumerate(JOINT_ORDER):
        # 左列：位置对比
        ax_pos = axes[i, 0]
        ax_pos.plot(t, data['des_dof_pos'][:, i], 'b--', label='Desired', lw=2, alpha=0.8)
        ax_pos.plot(t, data['dof_pos'][:, i], 'r-', label='Actual', lw=1.5)

        # 计算跟踪误差
        error = (data['dof_pos'][:, i] - data['des_dof_pos'][:, i]).numpy()
        rmse = (error**2).mean()**0.5

        ax_pos.set_ylabel('Position [rad]')
        ax_pos.set_title(f'{name} - Position (RMSE: {rmse:.4f})', fontsize=10)
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc='best', fontsize=8)

        # 右列：速度（如果有数据）
        ax_vel = axes[i, 1]
        if has_velocity:
            vel = data['dof_vel'][:, i].numpy()
            ax_vel.plot(t, vel, 'g-', lw=1.5, label='Velocity')

            # 计算速度统计
            vel_mean = vel.mean()
            vel_std = vel.std()

            ax_vel.set_ylabel('Velocity [rad/s]')
            ax_vel.set_title(f'{name} - Velocity (μ={vel_mean:.3f}, σ={vel_std:.3f})', fontsize=10)
            ax_vel.axhline(y=0, color='k', linestyle='--', lw=0.8, alpha=0.5)
            ax_vel.legend(loc='best', fontsize=8)
        else:
            ax_vel.text(0.5, 0.5, 'No Velocity Data',
                        transform=ax_vel.transAxes, ha='center', va='center',
                        fontsize=12, color='gray')
            ax_vel.set_title(f'{name} - Velocity (No Data)', fontsize=10)
            ax_vel.grid(True, alpha=0.3)

    # 最后一行设置x轴标签
    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='将CSV文件转换为PyTorch .pt格式，并可视化轨迹',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换指定的CSV文件并绘图
  python %(prog)s --csvpath 260117_chrip_20s_3motors_aligned.csv

  # 转换指定CSV但不绘图
  python %(prog)s --csvpath 260117_sin5Hz_10s_3motors_aligned.csv --no-plot

  # 查看可用的CSV文件
  python %(prog)s --list
        """
    )

    parser.add_argument(
        '--csvpath',
        type=str,
        default=None,
        help='指定要转换的CSV文件（可以是文件名、相对路径或绝对路径）'
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='只转换文件，不显示图表'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用的CSV文件'
    )

    args = parser.parse_args()

    # 如果只是列出文件
    if args.list:
        csv_files = list(RAW_CSV_DIR.glob("*.csv"))
        if not csv_files:
            print("❌ 没找到任何CSV文件！")
            print(f"查找目录: {RAW_CSV_DIR.absolute()}")
        else:
            print(f"找到 {len(csv_files)} 个CSV文件:")
            for f in csv_files:
                print(f"  - {f.name}")
        return

    # 检查是否指定了CSV文件
    if not args.csvpath:
        print("❌ 请使用 --csvpath 指定要转换的CSV文件")
        print("\n使用 --list 查看可用的CSV文件")
        print("使用 --help 查看帮助信息")
        return

    # 处理CSV路径
    csv_path = Path(args.csvpath)

    # 如果是相对路径或只是文件名，在默认目录中查找
    if not csv_path.is_absolute():
        if csv_path.name == args.csvpath:  # 只提供了文件名
            csv_path = RAW_CSV_DIR / args.csvpath

    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        print("\n可用的CSV文件:")
        for f in RAW_CSV_DIR.glob("*.csv"):
            print(f"  - {f.name}")
        return

    print(f"指定文件: {csv_path.name}")

    # 转换文件
    pt_path = convert_csv_to_pt(csv_path)

    if not pt_path:
        print("❌ 转换失败")
        return

    print(f"\n✓ 转换成功: {pt_path.name}")

    # 绘图
    if not args.no_plot:
        print("\n绘制轨迹图...")
        plot_trajectory(pt_path)


if __name__ == "__main__":
    main()
