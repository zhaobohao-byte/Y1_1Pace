| 序号 | 关节名称 | 电机选型 | Armature (kg·m²) | 速度 (rad/s) | 力矩 (N·m) | 角度 (rad) |
| 18 | l_hip_pitch_joint | RS-03 | 0.02 | [-18.84, 18.84] | [-60, 60] | [-2.531, 2.880] |
| 19 | l_hip_roll_joint | RS-06 | 0.012 | [-10.5, 10.5] | [-36, 36] | [-0.262, 2.793] |
| 20 | l_hip_yaw_joint | RS-06 | 0.012 | [-10.5, 10.5] | [-36, 36] | [-2.793, 2.793] |
| 21 | l_knee_pitch_joint | RS-03 | 0.02 | [-18.84, 18.84] | [-60, 60] | [-0.087, 2.793] |
| 22 | l_ankle_pitch_joint | RS-06 | 0.012 | [-10.5, 10.5] | [-36, 36] | [-0.873, 0.346] |
| 23 | l_ankle_roll_joint | RS-00 | 0.001 | [-27.2, 27.2] | [-14, 14] | [-0.346, 0.346] |

# 说明
- **ARMATURE_RS_06** = 0.012 kg·m² (用于: waist_yaw, hip_roll, hip_yaw, ankle_pitch)
- **ARMATURE_RS_03** = 0.02 kg·m² (用于: hip_pitch, knee)
- **ARMATURE_RS_00** = 0.001 kg·m² (用于: ankle_roll, 所有上肢关节)