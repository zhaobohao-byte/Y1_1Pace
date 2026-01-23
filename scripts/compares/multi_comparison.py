#!/usr/bin/env python3
"""
多文件比较工具 - 位置和速度数据对比

Usage:
    python multi_comparison.py file1.pt file2.pt [file3.pt] [-o output.png]
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_data(file_path):
    """加载数据文件"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    data = torch.load(file_path)
    result = {'name': file_path.stem, 'path': file_path}

    # 保留所有关节数据（不只取第一个关节）
    for key in ['time', 'dof_pos', 'dof_vel', 'des_dof_pos']:
        if key in data:
            tensor = data[key]
            if key == 'time':
                result[key] = tensor.numpy() if tensor.ndim == 1 else tensor.numpy()
            else:
                # 保留所有关节数据 (time_steps, n_joints)
                result[key] = tensor.numpy() if tensor.ndim > 1 else tensor.numpy().reshape(-1, 1)

    # 获取关节数量
    if 'dof_pos' in result:
        result['n_joints'] = result['dof_pos'].shape[1] if result['dof_pos'].ndim > 1 else 1

    return result


def plot_comparison(data_list, output_path):
    """绘制所有关节的比较图
    
    颜色方案：
    - 期望位置（target）: 灰色虚线
    - 仿真位置/速度: 蓝色虚线
    - 真实位置/速度: 红色实线
    """
    labels = [d['name'] for d in data_list]

    # 获取关节数量
    n_joints = data_list[0].get('n_joints', 1)
    print(f"\n检测到 {n_joints} 个关节")

    # 为每个关节创建一个图
    for joint_idx in range(n_joints):
        print(f"  绘制关节 {joint_idx + 1}/{n_joints}...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Joint {joint_idx + 1} Comparison', fontsize=16, fontweight='bold', y=0.995)

        # 位置比较
        for i, data in enumerate(data_list):
            # 绘制实际位置数据
            pos_data = data['dof_pos'][:, joint_idx] if data['dof_pos'].ndim > 1 else data['dof_pos']
            
            if i == 0:
                # 第一个文件（真实数据）：红色实线
                ax1.plot(data['time'], pos_data,
                         color='red', linestyle='-', linewidth=2,
                         alpha=0.9, zorder=3, label=f'{labels[i]} (Real)')
            else:
                # 其他文件（仿真数据）：蓝色虚线
                ax1.plot(data['time'], pos_data,
                         color='blue', linestyle='--', linewidth=1.5,
                         alpha=0.8, zorder=2, label=f'{labels[i]} (Sim)')

            # 绘制期望位置（灰色虚线）
            if 'des_dof_pos' in data:
                des_pos_data = data['des_dof_pos'][:, joint_idx] if data['des_dof_pos'].ndim > 1 else data['des_dof_pos']
                ax1.plot(data['time'], des_pos_data,
                         color='gray', linestyle='--', linewidth=1.5,
                         alpha=0.6, zorder=1, label=f'{labels[i]} (Target)')

        ax1.set_xlabel('Time [s]', fontsize=12)
        ax1.set_ylabel('Position [rad]', fontsize=12)
        ax1.set_title(f'Joint {joint_idx + 1} - Position Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(alpha=0.3)

        # 速度比较
        has_velocity = any('dof_vel' in d for d in data_list)
        if has_velocity:
            for i, data in enumerate(data_list):
                if 'dof_vel' in data:
                    # 绘制速度数据
                    vel_data = data['dof_vel'][:, joint_idx] if data['dof_vel'].ndim > 1 else data['dof_vel']
                    
                    if i == 0:
                        # 第一个文件（真实数据）：红色实线
                        ax2.plot(data['time'], vel_data,
                                 color='red', linestyle='-', linewidth=2,
                                 alpha=0.9, zorder=3, label=f'{labels[i]} (Real)')
                    else:
                        # 其他文件（仿真数据）：蓝色虚线
                        ax2.plot(data['time'], vel_data,
                                 color='blue', linestyle='--', linewidth=1.5,
                                 alpha=0.8, zorder=2, label=f'{labels[i]} (Sim)')

            ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
            ax2.set_xlabel('Time [s]', fontsize=12)
            ax2.set_ylabel('Velocity [rad/s]', fontsize=12)
            ax2.set_title(f'Joint {joint_idx + 1} - Velocity Comparison', fontsize=14, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No velocity data available',
                     transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title(f'Joint {joint_idx + 1} - Velocity Comparison (No Data)', fontsize=14, fontweight='bold')

        plt.tight_layout()

        # 保存每个关节的图
        if n_joints > 1:
            joint_output_path = output_path.parent / f"{output_path.stem}_joint{joint_idx + 1}{output_path.suffix}"
        else:
            joint_output_path = output_path

        plt.savefig(joint_output_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ 已保存: {joint_output_path.name}")
        plt.show()

    print(f"\n✓ 所有比较图已保存到: {output_path.parent}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多文件比较工具 - 位置和速度数据对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 比较2个文件
  python %(prog)s file1.pt file2.pt

  # 比较3个文件
  python %(prog)s file1.pt file2.pt file3.pt

  # 指定输出路径
  python %(prog)s file1.pt file2.pt -o comparison.png

  # 使用绝对路径
  python %(prog)s /path/to/file1.pt /path/to/file2.pt
        """
    )
    parser.add_argument('files', nargs='*',  # 改为 '*' 允许0个或多个参数
                        default=[
                            "/home/bohao/LuvRobot/Y1_1Pace/data/RS_motors/raw_pt/RS06_step_10s_aligned.pt",
                            "/home/bohao/LuvRobot/Y1_1Pace/data/sim_data/RS06_step_10s_aligned_sim_output_pv_150.pt"
                        ],
                        help='要比较的PT文件的绝对路径 (2-4个文件，不提供则使用默认文件)')
    parser.add_argument('-o', '--output',
                        default="/home/bohao/LuvRobot/Y1_1Pace/scripts/compares/comparison.png",
                        help='输出图片路径')
    args = parser.parse_args()

    # 验证文件数量
    if len(args.files) < 2:
        parser.error("至少需要2个文件进行比较")
    if len(args.files) > 4:
        parser.error("最多支持4个文件比较")

    # 转换为Path对象
    file_paths = [Path(f) for f in args.files]

    # 验证文件存在性
    for file_path in file_paths:
        if not file_path.exists():
            parser.error(f"文件不存在: {file_path}")

    print("\n" + "=" * 70)
    print("多文件比较工具")
    print("=" * 70)

    # 加载数据
    data_list = []
    for file_path in file_paths:
        try:
            data = load_data(file_path)
            data_list.append(data)
            print(f"✓ 已加载: {file_path.name}")
        except Exception as e:
            print(f"✗ 加载失败 {file_path.name}: {e}")
            return

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        # 默认保存在第一个文件所在目录
        output_dir = file_paths[0].parent
        # 生成文件名
        names = [d['name'] for d in data_list]
        output_filename = f"comparison_{'_'.join(name[:15] for name in names[:2])}.png"
        output_path = output_dir / output_filename

    # 生成比较图
    print("\n" + "=" * 70)
    print("生成比较图...")
    print("=" * 70)
    plot_comparison(data_list, output_path)

    print("\n" + "=" * 70)
    print("比较完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
