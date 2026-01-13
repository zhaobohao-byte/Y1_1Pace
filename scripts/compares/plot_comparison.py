#!/usr/bin/env python3
"""Plot comparison between real and simulation data"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys


def plot_comparison_with_paths(real_data_path, sim_data_path, output_prefix=None):
    """
    绘制真实数据和仿真数据的对比图（使用完整路径）

    Args:
        real_data_path: 真实数据文件的完整路径（Path对象或字符串）
        sim_data_path: 仿真数据文件的完整路径（Path对象或字符串）
        output_prefix: 输出图片文件名前缀，默认根据real_data_file生成
    """
    real_data_path = Path(real_data_path)
    sim_data_path = Path(sim_data_path)
    
    # Load real data
    if not real_data_path.exists():
        raise FileNotFoundError(f"Real data file not found: {real_data_path}")

    real_data = torch.load(real_data_path)
    time_real = real_data["time"].numpy()
    input_real = real_data["des_dof_pos"][:, 0].numpy()  # Input (u)
    output_real = real_data["dof_pos"][:, 0].numpy()  # Real output (y)

    # Load simulation data
    if not sim_data_path.exists():
        raise FileNotFoundError(f"Simulation data file not found: {sim_data_path}")

    sim_data = torch.load(sim_data_path)
    time_sim = sim_data["time"].numpy()
    output_sim = sim_data["dof_pos"][:, 0].numpy()  # Simulation output
    
    # 确定输出目录（使用真实数据所在的目录）
    output_dir = real_data_path.parent

    print(f"Real data: time range {time_real[0]:.4f} - {time_real[-1]:.4f} s, {len(time_real)} points")
    print(f"Simulation data: time range {time_sim[0]:.4f} - {time_sim[-1]:.4f} s, {len(time_sim)} points")

    # Calculate errors
    tracking_error_real = output_real - input_real
    tracking_error_sim = output_sim - input_real
    output_diff = output_sim - output_real

    rmse_real = np.sqrt(np.mean(tracking_error_real ** 2))
    rmse_sim = np.sqrt(np.mean(tracking_error_sim ** 2))
    rmse_diff = np.sqrt(np.mean(output_diff ** 2))
    max_diff = np.abs(output_diff).max()

    # 确定输出文件名前缀
    if output_prefix is None:
        output_prefix = real_data_path.stem.replace('_data', '')

    # Figure 1: Input + Real Output + Simulation Output
    plt.figure(figsize=(12, 6))
    plt.plot(time_real, input_real, 'k--', label='Input (u)', linewidth=1, alpha=0.7)
    plt.plot(time_real, output_real, 'b-', label='Real Output (y_real)', linewidth=1)
    plt.plot(time_sim, output_sim, 'r-', label='Simulation Output (y_sim)', linewidth=1, alpha=0.8)
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Position [rad]', fontsize=12)
    plt.title('Real vs Simulation Output', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure 1
    output_path1 = output_dir / f"{output_prefix}_comparison.png"
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_path1}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Comparison Statistics")
    print("=" * 70)
    print("\nReal Data Tracking Performance:")
    print(f"  RMSE: {rmse_real:.6f} rad")
    print(f"  Max Error: {np.abs(tracking_error_real).max():.6f} rad")

    print("\nSimulation Data Tracking Performance:")
    print(f"  RMSE: {rmse_sim:.6f} rad")
    print(f"  Max Error: {np.abs(tracking_error_sim).max():.6f} rad")

    print("\nSimulation vs Real Difference:")
    print(f"  RMSE: {rmse_diff:.6f} rad")
    print(f"  Max Difference: {max_diff:.6f} rad")
    print(f"  Mean Difference: {np.mean(output_diff):.6f} rad")
    print(f"  Std Deviation: {np.std(output_diff):.6f} rad")
    print("\n" + "=" * 70)

    plt.show()


def plot_comparison(real_data_file, sim_data_file, output_prefix=None):
    """
    绘制真实数据和仿真数据的对比图（向后兼容版本）

    Args:
        real_data_file: 真实数据文件名（在data/DM8006/目录下）
        sim_data_file: 仿真数据文件名（在data/DM8006/目录下）
        output_prefix: 输出图片文件名前缀，默认根据real_data_file生成
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "DM8006"
    real_data_path = data_dir / real_data_file
    sim_data_path = data_dir / sim_data_file
    plot_comparison_with_paths(real_data_path, sim_data_path, output_prefix)


def select_from_menu(options, title="Please select:"):
    """
    交互式菜单选择
    
    Args:
        options: 选项列表
        title: 菜单标题
    
    Returns:
        选中的选项
    """
    try:
        import inquirer
        questions = [
            inquirer.List('selection',
                          message=title,
                          choices=options,
                          ),
        ]
        answers = inquirer.prompt(questions)
        return answers['selection'] if answers else None
    except ImportError:
        # 如果没有inquirer，使用简单的数字选择
        print(f"\n{title}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        while True:
            try:
                choice = input("\nEnter number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
                else:
                    print(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nCancelled")
                return None


def ():
    """交互式模式"""
    base_data_dir = Path(__file__).parent.parent.parent / "data" / "DM8006"
    
    print("\n" + "="*70)
    print("Interactive Comparison Plot Tool")
    print("="*70)
    
    # 获取所有子文件夹（包括当前目录）
    subdirs = [d for d in base_data_dir.iterdir() if d.is_dir()]
    # 添加根目录选项
    dir_options = ["DM8006 (root)"] + [d.name for d in subdirs]
    
    # 选择真实数据所在的文件夹
    print("\nStep 1: Select folder for REAL data")
    real_folder_choice = select_from_menu(dir_options, "Select folder containing REAL data:")
    if not real_folder_choice:
        print("Cancelled")
        return
    
    # 确定真实数据目录
    if real_folder_choice == "DM8006 (root)":
        real_data_dir = base_data_dir
    else:
        real_data_dir = base_data_dir / real_folder_choice
    
    # 获取真实数据文件夹中的PT文件
    real_pt_files = sorted([f.name for f in real_data_dir.glob("*.pt")])
    if not real_pt_files:
        print(f"No .pt files found in {real_data_dir}")
        return
    
    # 选择真实数据文件
    print(f"\nFolder: {real_data_dir.relative_to(base_data_dir.parent)}")
    real_data_file = select_from_menu(real_pt_files, "Select REAL data file:")
    if not real_data_file:
        print("Cancelled")
        return
    
    real_data_path = real_data_dir / real_data_file
    
    # 选择仿真数据所在的文件夹
    print(f"\nSelected real data: {real_data_path.relative_to(base_data_dir.parent)}")
    print("\nStep 2: Select folder for SIMULATION data")
    sim_folder_choice = select_from_menu(dir_options, "Select folder containing SIMULATION data:")
    if not sim_folder_choice:
        print("Cancelled")
        return
    
    # 确定仿真数据目录
    if sim_folder_choice == "DM8006 (root)":
        sim_data_dir = base_data_dir
    else:
        sim_data_dir = base_data_dir / sim_folder_choice
    
    # 获取仿真数据文件夹中的PT文件
    sim_pt_files = sorted([f.name for f in sim_data_dir.glob("*.pt")])
    if not sim_pt_files:
        print(f"No .pt files found in {sim_data_dir}")
        return
    
    # 选择仿真数据文件
    print(f"\nFolder: {sim_data_dir.relative_to(base_data_dir.parent)}")
    sim_data_file = select_from_menu(sim_pt_files, "Select SIMULATION data file:")
    if not sim_data_file:
        print("Cancelled")
        return
    
    sim_data_path = sim_data_dir / sim_data_file
    
    print(f"\nSelected simulation data: {sim_data_path.relative_to(base_data_dir.parent)}")
    
    # 询问是否自定义输出前缀
    custom_prefix = input("\nCustom output prefix (press Enter to auto-generate): ").strip()
    output_prefix = custom_prefix if custom_prefix else None
    
    print("\n" + "="*70)
    print("Generating comparison plot...")
    print("="*70)
    
    # 绘制对比图（传入完整路径）
    try:
        plot_comparison_with_paths(real_data_path, sim_data_path, output_prefix)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制真实数据和仿真数据的对比图")
    parser.add_argument("--real_data", type=str, default=None,
                        help="真实数据文件名（在data/DM8006/目录下）")
    parser.add_argument("--sim_data", type=str, default=None,
                        help="仿真数据文件名（在data/DM8006/目录下）")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="输出图片文件名前缀，默认自动生成")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互式模式")

    args = parser.parse_args()
    
    # 如果没有提供参数或指定了交互模式，进入交互模式
    if args.interactive or (args.real_data is None and args.sim_data is None):
        interactive_mode()
    else:
        if args.real_data is None or args.sim_data is None:
            parser.error("--real_data and --sim_data are required in non-interactive mode")
        plot_comparison(args.real_data, args.sim_data, args.output_prefix)
