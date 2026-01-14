#!/usr/bin/env python3
"""
CSV to PyTorch PT File Converter

将 CSV 文件转换为 PyTorch .pt 格式的交互式工具。

依赖: torch, pandas, matplotlib, numpy

用法: python csv2pt.py
"""

import sys
from pathlib import Path

# 检查依赖
deps = ["torch", "pandas", "matplotlib", "numpy"]
missing = []
for dep in deps:
    try:
        __import__(dep)
    except ImportError:
        missing.append(dep)

if missing:
    print(f"缺少依赖: {' '.join(missing)}")
    print(f"安装命令: pip install {' '.join(missing)}")
    sys.exit(1)

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def select_from_menu(options, title="请选择:"):
    """交互式菜单选择"""
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")

    while True:
        try:
            choice = input("\n输入数字 (q退出): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            return options[idx] if 0 <= idx < len(options) else None
        except (ValueError, KeyboardInterrupt):
            return None

def find_csv_files(directory):
    """递归查找CSV文件"""
    return [(str(p.relative_to(directory)), p) for p in sorted(directory.rglob("*.csv"))]

def main():
    """主函数"""
    # 查找项目根目录
    cwd = Path.cwd()
    project_root = next((p for p in [cwd] + list(cwd.parents)
                        if (p / "data").exists() and (p / "source").exists()), None)
    if not project_root:
        print("错误: 找不到项目根目录")
        return

    data_dir = project_root / "data"
    print("=" * 50)
    print("CSV 转 PT 工具")
    print("=" * 50)

    # 选择数据文件夹
    subdirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("错误: data目录下无子文件夹")
        return

    folder = select_from_menu(subdirs, "选择数据文件夹:")
    if not folder:
        return

    selected_folder = data_dir / folder
    print(f"已选择: {folder}")

    # 选择CSV文件
    csv_files = find_csv_files(selected_folder)
    if not csv_files:
        print(f"错误: {selected_folder} 下无CSV文件")
        return

    csv_options = [f"{name} ({path.parent.name})" if path.parent != selected_folder else name
                   for name, path in csv_files]
    selected = select_from_menu(csv_options, "选择CSV文件:")
    if not selected:
        return

    csv_path = csv_files[csv_options.index(selected)][1]

    # 读取并转换数据
    print(f"\n读取: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"形状: {df.shape}, 列: {list(df.columns)}")

    # 转换为tensor
    data = {k: torch.tensor(df[c].values, dtype=torch.float32).unsqueeze(-1)
            for k, c in [("time", "time"), ("dof_pos", "y"), ("des_dof_pos", "u"), ("dof_vel", "dy")]}
    print(f"数据shapes: {[(k, v.shape) for k, v in data.items()]}")

    # 选择输出目录
    print("\n" + "=" * 50)
    print("选择输出位置")
    print("=" * 50)

    options = [f"当前文件夹 ({folder})", "其他文件夹", "新建文件夹"]
    choice = select_from_menu(options, "选择输出位置:")
    if not choice:
        return

    if "当前文件夹" in choice:
        output_dir = selected_folder
    elif "其他文件夹" in choice:
        others = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name != folder]
        out_folder = select_from_menu(others, "选择输出文件夹:") if others else None
        output_dir = data_dir / out_folder if out_folder else selected_folder
    else:
        name = input("新文件夹名: ").strip()
        output_dir = data_dir / name if name else selected_folder

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存文件
    pt_path = output_dir / f"{csv_path.stem}.pt"
    torch.save(data, pt_path)
    print(f"\n保存: {pt_path} ({pt_path.stat().st_size / 1024:.1f}KB)")

    # 生成对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    t = data["time"].numpy()

    # 位置对比
    ax1.plot(t, data["des_dof_pos"][:, 0].numpy(), color='lightgray', label='Target', linewidth=1, alpha=0.7)
    ax1.plot(t, data["dof_pos"][:, 0].numpy(), 'b-', label='Actual', linewidth=1)
    ax1.set_xlabel('Time [s]'), ax1.set_ylabel('Position [rad]')
    ax1.set_title('Position Comparison'), ax1.legend(), ax1.grid(alpha=0.3)

    # 速度对比
    ax2.plot(t, data["dof_vel"][:, 0].numpy(), 'r-', label='Velocity', linewidth=1)
    ax2.set_xlabel('Time [s]'), ax2.set_ylabel('Velocity [rad/s]')
    ax2.set_title('Velocity'), ax2.legend(), ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"{csv_path.stem}_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"图表: {plot_path}")

    print("\n" + "=" * 50)
    print("转换完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()