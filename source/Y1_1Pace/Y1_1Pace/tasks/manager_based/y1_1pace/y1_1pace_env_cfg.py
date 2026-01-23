from isaaclab.utils import configclass
# from isaaclab.sim import sim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from Y1_1Pace.utils import PaceDCMotorCfg, project_root
from Y1_1Pace.utils import PaceImplicitActuatorCfg
from Y1_1Pace import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch
import os


################################################################################
#  RS06
################################################################################
# RS06_PACE_ACTUATOR_CFG = PaceDCMotorCfg(               # use DC motor model
RS06_PACE_ACTUATOR_CFG = PaceImplicitActuatorCfg(         # use implicit actuator model
    joint_names_expr=["waist_yaw_joint"],      
    armature={
        "waist_yaw_joint": 0.012,
    },
    stiffness={
        "waist_yaw_joint": 47.3740,
    },
    damping={
        "waist_yaw_joint": 3.0160,
    },
)

@configclass
class RS06PaceCfg(PaceCfg):
    """Pace configuration for Atom3DOF robot."""
    robot_name: str = "Atom3DOF_sim"
    data_dir: str = "RS_motors/raw_pt/RS06_chrip_spd10_aligned.pt"  # located in Y1_1Pace/data/Atom3DOF_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((3, 2))  # 1 + 1 + 1 = 3 parameters to optimize
    joint_order: list[str] = [
        "waist_yaw_joint",
    ]

    def __post_init__(self):
        # set bounds for parameters
        # 参数顺序: [armature, damping, friction]
        self.bounds_params[0, 0] = 1e-5        # armature 下界
        self.bounds_params[0, 1] = 0.8         # armature 上界
        self.bounds_params[1, 1] = 10.0        # damping 上界
        self.bounds_params[2, 0] = 0.0         # friction 下界
        self.bounds_params[2, 1] = 1.0         # friction 上界

################################################################################
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
                "RS06",
                "urdf",
                "single_joint.urdf"
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
            "waist_yaw_joint": RS06_PACE_ACTUATOR_CFG
        },
    )



@configclass
class Y1_1PaceEnvCfg(PaceSim2realEnvCfg):

    scene: Y1_1PaceSceneCfg = Y1_1PaceSceneCfg()
    # sim2real: PaceCfg = Y1_1PaceCfg()
    sim2real: PaceCfg = RS06PaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.002  # 500Hz simulation
        self.decimation = 1  # 500Hz control
