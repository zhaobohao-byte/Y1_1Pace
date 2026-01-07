# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass
from isaaclab.sim import sim_utils
from isaaclab_assets.robots.Y1_1 import Y1_1_D_CFG
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

ANYDRIVE_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=140.0,
    effort_limit=89.0,
    velocity_limit=8.5,
    stiffness={".*": 85.0},  # P gain in Nm/rad
    damping={".*": 0.6},  # D gain in Nm s/rad
    encoder_bias=[0.0] * 12,  # encoder bias in radians
    max_delay=10,  # max delay in simulation steps
)


@configclass
class Y1_1DPaceCfg(PaceCfg):
    """Pace configuration for Y1_1-D robot."""
    robot_name: str = "Y1_1_d_sim"
    data_dir: str = "Y1_1_d_sim/chirp_data.pt"  # located in pace_sim2real/data/Y1_1_d_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49 parameters to optimize
    joint_order: list[str] = [
        "LF_HAA",
        "LF_HFE",
        "LF_KFE",
        "RF_HAA",
        "RF_HFE",
        "RF_KFE",
        "LH_HAA",
        "LH_HFE",
        "LH_KFE",
        "RH_HAA",
        "RH_HFE",
        "RH_KFE",
    ]






    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:12, 0] = 1e-5
        self.bounds_params[:12, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[12:24, 1] = 7.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[24:36, 1] = 0.5  # friction between 0.0 - 0.5
        self.bounds_params[36:48, 0] = -0.1
        self.bounds_params[36:48, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[48, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class Y1_1DPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Y1_1-D robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = Y1_1_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
                                                  actuators={"legs": ANYDRIVE_PACE_ACTUATOR_CFG})

Y1_1_LEFTLEG_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",          
    spawn=sim_utils.UrdfFileCfg(        
        urdf_path="/path/to/your/atomV3_leftleg.urdf",  # 修改成你的 URDF 文件完整路径
        
        # 可选：如果 URDF 中的 mesh 文件是相对路径，这里指定资产根目录
        # asset_root_path="/path/to/your/meshes_directory",  # 如果需要
        
        # 其他常用参数（类似 UsdFileCfg）
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,                    # floating-base，必须 False
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        # 可选：覆盖 URDF 中的视觉材质（如果需要统一外观）
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.9)),
        
        # 可选：如果 URDF 转换有问题，可以强制每次都重新转换
        # force_usd_conversion=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),                    # 与你的 MJCF/URDF base_link pos 一致
        rot=(1.0, 0.0, 0.0, 0.0),               # 四元数 (w, x, y, z)
        joint_pos={
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_hip_pitch_joint": 0.0,
            "left_knee_joint": 1.0,             # 示例：膝盖稍弯曲，更稳定
            "left_ankle_joint": 0.0,
            # 手臂关节示例（根据你的 URDF 关节名添加）
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            # right_xxx 同理
        },
    ),
    actuators={
        "legs": ActuatorNetPDCfg(               # 推荐使用这个，更接近真实电机
            joint_names_expr=["left_.*_joint"], # 正则匹配左腿所有关节
            effort_limit=25.0,                  # 根据你的电机 torque limit 调整
            velocity_limit=20.0,                # rad/s
            stiffness=150.0,                    # 位置刚度 (position stiffness)
            damping=5.0,                        # 速度阻尼
        ),
        "arms": ActuatorNetPDCfg(
            joint_names_expr=[".*shoulder_.*", ".*elbow_.*"],
            effort_limit=10.0,
            velocity_limit=15.0,
            stiffness=100.0,
            damping=3.0,
        ),
        # 如果有 right 腿或其他组，可继续添加
    },
)





@configclass
class Y1_1DPaceEnvCfg(PaceSim2realEnvCfg):

    scene: Y1_1DPaceSceneCfg = Y1_1DPaceSceneCfg()
    sim2real: PaceCfg = Y1_1DPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.002  # 500Hz simulation
        self.decimation = 1  # 500Hz control
