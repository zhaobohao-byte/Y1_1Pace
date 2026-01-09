from isaaclab.utils import configclass
# from isaaclab.sim import sim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from Y1_1Pace.utils import PaceDCMotorCfg, project_root
from Y1_1Pace import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch
import os


Y1_1_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=["l_.*_joint"],      
    saturation_effort=10.0,
    effort_limit={
        "l_hip_yaw_joint": 5.0,
    },
    velocity_limit={
        "l_hip_yaw_joint": 10.0,
    },
    velocity_limit_sim={
        "l_hip_yaw_joint": 4.0,
    },
    armature={
        # "l_hip_yaw_joint": 0.012,       # RS-06
        "l_hip_yaw_joint": 1.0,
    },

    stiffness={
        # "l_hip_yaw_joint": 47.3741,
        "l_hip_yaw_joint": 0,
    },
    damping={
        # "l_hip_yaw_joint": 3.01592894736,
        "l_hip_yaw_joint": 0,
    },
    # encoder_bias=[0.0] * 6,  
    encoder_bias=[0.0],
    max_delay=1,  
)


@configclass
class Y1_1PaceCfg(PaceCfg):
    """Pace configuration for Y1_1 robot."""
    robot_name: str = "Y1_1_sim"
    data_dir: str = "Y1_1_sim/chrip_data_mujoco_noise.pt"  # located in Y1_1Pace/data/Y1_1_sim/chirp_data.pt
    # bounds_params: torch.Tensor = torch.zeros((25, 2))  # 6 + 6 + 6 + 6 + 1 = 25 parameters to optimize
    bounds_params: torch.Tensor = torch.zeros((5, 2))  # 1 = 1 parameters to optimize
    joint_order: list[str] = [
        "l_hip_pitch_joint",
        "l_hip_roll_joint",
        "l_hip_yaw_joint",
        "l_knee_pitch_joint",
        "l_ankle_pitch_joint",
        "l_ankle_roll_joint",
    ]

    def __post_init__(self):
        # set bounds for parameters
        # bounds_params shape: (5, 2) where each row is [lower_bound, upper_bound]
        # Index 0: armature
        self.bounds_params[0, 0] = 1e-5        # armature lower bound
        self.bounds_params[0, 1] = 0.8         # armature upper bound
        # Index 1: dof_damping
        self.bounds_params[1, 0] = 0.0         # dof_damping lower bound
        self.bounds_params[1, 1] = 10.0        # dof_damping upper bound
        # Index 2: friction
        self.bounds_params[2, 0] = 0.0         # friction lower bound
        self.bounds_params[2, 1] = 0.50         # friction upper bound
        # Index 3: bias
        self.bounds_params[3, 0] = -0.1         # bias lower bound
        self.bounds_params[3, 1] = 0.1          # bias upper bound
        # Index 4: delay
        self.bounds_params[4, 0] = 0.0         # delay lower bound
        self.bounds_params[4, 1] = 10.0         # delay upper bound

@configclass
class Y1_1PaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Y1_1 robot in Pace Sim2Real environment.
    """

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg( 
            asset_path = os.path.join(
                project_root(),
                "source",
                "Y1_1Pace",
                "Y1_1Pace",
                "tasks",
                "manager_based",
                "Y1_1_robot",
                "urdf",
                "Y1_1_Link.urdf"
            ), 
            fix_base=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(            
                enabled_self_collisions=False,

            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None,  # 从 URDF 读取
                    damping=None     # 从 URDF 读取或默认
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),                    
            rot=(1.0, 0.0, 0.0, 0.0),                 
        ),
        actuators={
            "leg_motors": Y1_1_PACE_ACTUATOR_CFG
        },
    )



@configclass
class Y1_1PaceEnvCfg(PaceSim2realEnvCfg):

    scene: Y1_1PaceSceneCfg = Y1_1PaceSceneCfg()
    sim2real: PaceCfg = Y1_1PaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.002  # 500Hz simulation
        self.decimation = 1  # 500Hz control
