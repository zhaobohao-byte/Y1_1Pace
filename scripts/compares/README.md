# 数据对比流程说明

本目录包含用于比较真实数据和仿真数据的通用脚本。

## 工作流程

### 1. 转换CSV数据为PT格式

```bash
# 转换5Hz正弦波数据
python scripts/utils/convert_csv_to_pt.py \
  --csv 20250112_5Hzsine_DM8006_500Hz_aligned.csv \
  --output 5Hz_sine_data

# 转换steps数据
python scripts/utils/convert_csv_to_pt.py \
  --csv 20250112_steps_DM8006_500Hz_aligned.csv \
  --output steps_data
```

参数说明：
- `--csv`: CSV文件名（位于`data/DM8006/raw_data/`目录）
- `--output`: 输出PT文件名（不含.pt后缀），默认自动生成

### 2. 运行仿真收集数据

```bash
cd /path/to/IsaacLab

# 收集5Hz正弦波的仿真数据
./isaaclab.sh -p /path/to/Y1_1Pace/scripts/compares/comp_collection.py \
  --headless \
  --input_data 5Hz_sine_data.pt \
  --output_suffix sim_output

# 收集steps的仿真数据
./isaaclab.sh -p /path/to/Y1_1Pace/scripts/compares/comp_collection.py \
  --headless \
  --input_data steps_data.pt \
  --output_suffix sim_output
```

参数说明：
- `--input_data`: 输入轨迹数据文件（位于`data/DM8006/`目录）
- `--output_suffix`: 输出文件后缀，生成`<input_name>_<suffix>.pt`
- `--headless`: 无头模式运行

### 3. 绘制对比图

```bash
# 绘制5Hz正弦波对比图
python scripts/compares/plot_comparison.py \
  --real_data 5Hz_sine_data.pt \
  --sim_data 5Hz_sine_data_sim_output.pt \
  --output_prefix 5Hz_sine

# 绘制steps对比图
python scripts/compares/plot_comparison.py \
  --real_data steps_data.pt \
  --sim_data steps_data_sim_output.pt \
  --output_prefix steps
```

参数说明：
- `--real_data`: 真实数据文件（位于`data/DM8006/`目录）
- `--sim_data`: 仿真数据文件（位于`data/DM8006/`目录）
- `--output_prefix`: 输出图片文件名前缀，默认自动生成

## 文件结构

```
data/DM8006/
├── raw_data/
│   ├── 20250112_5Hzsine_DM8006_500Hz_aligned.csv
│   └── 20250112_steps_DM8006_500Hz_aligned.csv
├── 5Hz_sine_data.pt                    # 转换后的真实数据
├── 5Hz_sine_data_sim_output.pt         # 仿真输出数据
├── 5Hz_sine_comparison.png             # 对比图
├── steps_data.pt                       # 转换后的真实数据
├── steps_data_sim_output.pt            # 仿真输出数据
└── steps_comparison.png                # 对比图
```

## 数据格式

所有PT文件包含以下字段：
- `time`: 时间序列 (1D tensor)
- `dof_pos`: 实际关节位置 (2D tensor, shape: [num_steps, num_joints])
- `des_dof_pos`: 目标关节位置 (2D tensor, shape: [num_steps, num_joints])

## 快速示例

```bash
# 完整流程示例（针对steps数据）
# 1. 转换CSV
python scripts/utils/convert_csv_to_pt.py \
  --csv 20250112_steps_DM8006_500Hz_aligned.csv \
  --output steps_data

# 2. 运行仿真（需要在IsaacLab环境中）
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/Y1_1Pace/scripts/compares/comp_collection.py \
  --headless \
  --input_data steps_data.pt

# 3. 绘制对比图
python scripts/compares/plot_comparison.py \
  --real_data steps_data.pt \
  --sim_data steps_data_sim_output.pt
```
