#!/usr/bin/env python3
"""将CSV数据转换为PT格式（通用脚本）"""

import torch
import pandas as pd
from pathlib import Path
import argparse

def convert_csv_to_pt(csv_filename, output_name=None):
    """
    将CSV数据转换为PT格式
    
    Args:
        csv_filename: CSV文件名（在data/DM8006/raw_data/目录下）
        output_name: 输出文件名（不含.pt后缀），默认与CSV文件名相同
    """
    # 路径设置
    base_dir = Path(__file__).parent.parent.parent / "data" / "DM8006"
    csv_path = base_dir / "raw_data" / csv_filename
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    print(f"CSV文件: {csv_filename}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"数据统计:")
    print(df.describe())
    
    # 转换为tensor
    time_data = torch.tensor(df['time'].values, dtype=torch.float32)
    des_dof_pos = torch.tensor(df['u'].values, dtype=torch.float32)
    dof_pos = torch.tensor(df['y'].values, dtype=torch.float32)
    
    # 添加维度 (num_steps, 1)
    des_dof_pos = des_dof_pos.unsqueeze(1)
    dof_pos = dof_pos.unsqueeze(1)
    
    print(f"\ntime shape: {time_data.shape}")
    print(f"des_dof_pos shape: {des_dof_pos.shape}")
    print(f"dof_pos shape: {dof_pos.shape}")
    
    # 确定输出文件名
    if output_name is None:
        # 从CSV文件名提取（移除_DM8006_500Hz_aligned后缀）
        output_name = csv_filename.replace('_DM8006_500Hz_aligned.csv', '_data')
        output_name = output_name.replace('.csv', '')
    
    # 保存为PT文件
    output_path = base_dir / f"{output_name}.pt"
    torch.save({
        "time": time_data,
        "dof_pos": dof_pos,
        "des_dof_pos": des_dof_pos,
    }, output_path)
    
    print(f"\n✓ 数据已保存到: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将CSV数据转换为PT格式")
    parser.add_argument("--csv", type=str, 
                       default="20250112_5Hzsine_DM8006_500Hz_aligned.csv",
                       help="CSV文件名（在data/DM8006/raw_data/目录下）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件名（不含.pt后缀），默认自动生成")
    
    args = parser.parse_args()
    convert_csv_to_pt(args.csv, args.output)
