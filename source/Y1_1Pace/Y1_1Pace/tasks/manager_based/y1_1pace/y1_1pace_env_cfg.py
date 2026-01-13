from isaaclab.utils import configclass
# from isaaclab.sim import sim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from Y1_1Pace.utils import PaceDCMotorCfg, project_root
from Y1_1Pace import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG
import torch
import os


Y1_1_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=["l_.*_joint"],      
    saturation_effort=100.0,
    effort_limit={
        "l_hip_pitch_joint": 60.0,
        "l_hip_roll_joint": 36.0,
        "l_hip_yaw_joint": 36.0,
        "l_knee_pitch_joint": 60.0,
        "l_ankle_pitch_joint": 36.0,
        "l_ankle_roll_joint": 14.0,
    },
    velocity_limit={
        "l_hip_pitch_joint": 18.0,
        "l_hip_roll_joint": 10.0,
        "l_hip_yaw_joint": 10.0,
        "l_knee_pitch_joint": 18.0,
        "l_ankle_pitch_joint": 10.0,
        "l_ankle_roll_joint": 25.0,
    },
    armature={
        "l_hip_pitch_joint": 0.02,      # RS-03
        "l_hip_roll_joint": 0.012,      # RS-06
        "l_hip_yaw_joint": 0.012,       # RS-06
        "l_knee_pitch_joint": 0.02,     # RS-03
        "l_ankle_pitch_joint": 0.012,   # RS-06
        "l_ankle_roll_joint": 0.001,    # RS-00
    },

    stiffness={
        "l_hip_pitch_joint": 78.9568352,
        "l_hip_roll_joint": 47.3741,
        "l_hip_yaw_joint": 47.3741,
        "l_knee_pitch_joint": 78.9568352,    
        "l_ankle_pitch_joint": 47.3741,
        "l_ankle_roll_joint": 3.94784176,
    },
    damping={
        "l_hip_pitch_joint": 5.0265482456,
        "l_hip_roll_joint": 3.01592894736,
        "l_hip_yaw_joint": 3.01592894736,
        "l_knee_pitch_joint": 5.0265482456,
        "l_ankle_pitch_joint": 3.01592894736,
        "l_ankle_roll_joint": 0.25132741228,
    },
    encoder_bias=[0.0] * 6,  
    max_delay=10,  
)


@configclass
class Y1_1PaceCfg(PaceCfg):
    """Pace configuration for Y1_1 robot."""
    robot_name: str = "Y1_1_sim"
    data_dir: str = "Y1_1_sim/chrip_data_mujoco_noise.pt"  # located in Y1_1Pace/data/Y1_1_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((25, 2))  # 6 + 6 + 6 + 6 + 1 = 25 parameters to optimize
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
        self.bounds_params[:6, 0] = 1e-5        # armature 
        self.bounds_params[:6, 1] = 0.8        
        self.bounds_params[6:12, 1] = 10.0       # dof_damping
        self.bounds_params[12:18, 1] = 1.0      # friction
        self.bounds_params[18:24, 0] = -0.1     # bias
        self.bounds_params[18:24, 1] = 0.1     
        self.bounds_params[24, 1] = 10.0        # delay

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
                "Y1_1.urdf"
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
class AnymalDPaceCfg(PaceCfg):
    """Pace configuration for Anymal-D robot."""
    robot_name: str = "anymal_d_sim"
    data_dir: str = "anymal_d_sim/chrip_traj_data.pt"  # located in pace_sim2real/data/anymal_d_sim/chrip_data.pt
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
        self.bounds_params[:12, 1] = 2.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[12:24, 1] = 14.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[24:36, 1] = 1  # friction between 0.0 - 0.5
        self.bounds_params[36:48, 0] = -0.2
        self.bounds_params[36:48, 1] = 0.2  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[48, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]

@configclass
class ANYmalDPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Anymal-D robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
                                                  actuators={"legs": ANYDRIVE_PACE_ACTUATOR_CFG})
@configclass
class Y1_1PaceEnvCfg(PaceSim2realEnvCfg):

    # scene: Y1_1PaceSceneCfg = Y1_1PaceSceneCfg()
    # sim2real: PaceCfg = Y1_1PaceCfg()
    scene: ANYmalDPaceSceneCfg = ANYmalDPaceSceneCfg()
    sim2real: PaceCfg = AnymalDPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.002  # 500Hz simulation
        self.decimation = 1  # 500Hz control
        self.scene.robot.spawn.articulation_props.fix_root_link = True