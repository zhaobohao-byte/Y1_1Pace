#!/usr/bin/env python3
"""
多文件比较工具 - 位置和速度数据对比

Usage:
    python multi_comparison.py                    # 交互模式
    python multi_comparison.py file1.pt file2.pt  # 直接指定文件
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def select_pt_files(data_dir):
    """交互式选择PT文件"""
    pt_files = list(data_dir.rglob("*.pt"))
    if not pt_files:
        print(f"错误: {data_dir} 下无PT文件")
        return []

    print(f"\n找到 {len(pt_files)} 个PT文件:")
    for i, f in enumerate(pt_files, 1):
        print(f"  {i}. {f.relative_to(data_dir)}")

    selected = []
    while len(selected) < 3:
        try:
            choice = input(f"\n选择第{len(selected)+1}个文件 (数字/q退出): ").strip()
            if choice.lower() == 'q':
                break
            idx = int(choice) - 1
            if 0 <= idx < len(pt_files) and pt_files[idx] not in selected:
                selected.append(pt_files[idx])
                print(f"已选择: {pt_files[idx].name}")
            else:
                print("无效选择或重复选择")
        except (ValueError, KeyboardInterrupt):
            break
    return selected[:3]

def load_data(file_path):
    """加载数据文件"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    data = torch.load(file_path)
    result = {'name': file_path.stem}

    # 统一处理多维数据转换
    for key in ['time', 'dof_pos', 'dof_vel', 'des_dof_pos']:
        if key in data:
            tensor = data[key]
            result[key] = tensor[:, 0].numpy() if tensor.ndim > 1 else tensor.numpy()

    return result

def plot_comparison(data_list, output_path):
    """绘制比较图"""
    colors = ['gray', 'red', 'green']  # 第一个文件用灰色
    labels = [d['name'] for d in data_list]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 位置比较 - 第一个文件用虚线在底层
    for i, data in enumerate(data_list):
        style = {'linewidth': 1, 'zorder': -1 if i == 0 else 1}
        if i == 0:
            style.update({'color': 'gray', 'linestyle': '--', 'alpha': 0.7})

        ax1.plot(data['time'], data['dof_pos'], label=labels[i], **style)

        # 目标位置只为非第一个文件绘制
        if 'des_dof_pos' in data and i > 0:
            ax1.plot(data['time'], data['des_dof_pos'],
                    color=colors[i], linestyle='--', linewidth=1, alpha=0.7,
                    label=f"{labels[i]} (target)")

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position [rad]')
    ax1.set_title('Position Comparison')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 速度比较
    has_velocity = any('dof_vel' in d for d in data_list)
    if has_velocity:
        for i, data in enumerate(data_list):
            if 'dof_vel' in data:
                style = {'linewidth': 1, 'zorder': -1 if i == 0 else 1}
                if i == 0:
                    style.update({'color': 'gray', 'linestyle': '--', 'alpha': 0.7})
                else:
                    style['color'] = colors[i]
                ax2.plot(data['time'], data['dof_vel'], label=labels[i], **style)

        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Velocity [rad/s]')
        ax2.set_title('Velocity Comparison')
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No velocity data available',
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Velocity Comparison (No Data)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"比较图已保存: {output_path}")

def print_usage_example():
    """打印使用示例"""
    print("\n使用示例:")
    print("1. 交互模式: python multi_comparison.py")
    print("2. 指定文件: python multi_comparison.py file1.pt file2.pt")
    print("3. 指定输出: python multi_comparison.py file1.pt file2.pt -o output.png")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多文件比较工具")
    parser.add_argument('files', nargs='*', help='要比较的PT文件路径 (最多3个)')
    parser.add_argument('-o', '--output', help='输出图片路径')
    args = parser.parse_args()

    # 查找项目根目录
    cwd = Path.cwd()
    project_root = next((p for p in [cwd] + list(cwd.parents)
                        if (p / "data").exists() and (p / "source").exists()), None)
    if not project_root:
        print("错误: 找不到项目根目录 (需要包含 'data' 和 'source' 文件夹)")
        print_usage_example()
        return

    data_dir = project_root / "data"

    if args.files:
        # 命令行模式
        if len(args.files) < 2:
            print("错误: 至少需要2个文件进行比较")
            return

        selected_files = [Path(f) for f in args.files[:3]]  # 最多3个文件
        selected_dir = selected_files[0].parent  # 使用第一个文件所在目录
    else:
        # 交互模式
        print("=" * 60)
        print("多文件比较工具")
        print("=" * 60)

        # 选择数据文件夹
        subdirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
        if not subdirs:
            print("错误: data目录下无子文件夹")
            return

        print("\n选择数据文件夹:")
        for i, d in enumerate(subdirs, 1):
            print(f"  {i}. {d}")

        try:
            choice = input("\n输入文件夹编号: ").strip()
            idx = int(choice) - 1
            if not (0 <= idx < len(subdirs)):
                print("无效选择")
                return
            selected_dir = data_dir / subdirs[idx]
        except (ValueError, KeyboardInterrupt):
            return

        # 选择PT文件
        selected_files = select_pt_files(selected_dir)
        if len(selected_files) < 2:
            print("至少需要2个文件进行比较")
            return

    # 加载数据
    data_list = []
    for file_path in selected_files:
        try:
            data = load_data(file_path)
            data_list.append(data)
            print(f"已加载: {file_path.name}")
        except Exception as e:
            print(f"加载失败 {file_path.name}: {e}")
            return

    # 生成比较图
    if args.output:
        output_path = Path(args.output)
    else:
        # 智能文件名生成
        names = [d['name'] for d in data_list]
        prefix = names[0]
        for name in names[1:]:
            while prefix and not name.startswith(prefix):
                prefix = prefix[:-1]
        suffix = '_'.join(name[len(prefix):][:8] for name in names) if len(prefix) > 3 else '_'.join(name[:8] for name in names)
        output_path = selected_dir / f"comparison_{suffix}.png"

    plot_comparison(data_list, output_path)

    print("\n" + "=" * 60)
    print("比较完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()