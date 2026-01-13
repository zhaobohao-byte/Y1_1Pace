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
    # saturation_effort=78.22,      # RS-06
    saturation_effort=30,      # DM-8006
    effort_limit={
        # "l_hip_yaw_joint": 36.0,     # RS-06
        "l_hip_yaw_joint": 20,      # DM-8006
    },
    velocity_limit={
        # "l_hip_yaw_joint": 52.356,    # RS-06
        "l_hip_yaw_joint": 30.0,      # DM-8006
    },
    armature={
        # "l_hip_yaw_joint": 0.012,       # RS-06
        "l_hip_yaw_joint": 0.0126,      # DM-8006
    },
    stiffness={
        # "l_hip_yaw_joint": 47.3741,      # RS-06
        "l_hip_yaw_joint": 16.3440,              # DM-8006
    },
    damping={
        # "l_hip_yaw_joint": 3.01592894736,      # RS-06
        "l_hip_yaw_joint": 1.0400,              # DM-8006
    },
    # max_delay must be >= upper bound of delay parameter in bounds_params
    max_delay=10,
)


@configclass
class Y1_1PaceCfg(PaceCfg):
    """Pace configuration for Y1_1 robot."""
    robot_name: str = "Y1_1_sim"
    data_dir: str = "DM8006/chrip_data.pt"  # located in Y1_1Pace/data/Y1_1_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((4, 2))  # armature + damping + friction + delay = 4 parameters
    joint_order: list[str] = [
        "l_hip_yaw_joint",
    ]

    def __post_init__(self):
        # set bounds for parameters
        # bounds_params shape: (4, 2) where each row is [lower_bound, upper_bound]
        # Index 0: armature
        self.bounds_params[0, 0] = 1e-5        # armature lower bound
        self.bounds_params[0, 1] = 1           # armature upper bound
        # Index 1: dof_damping
        self.bounds_params[1, 1] = 7           # dof_damping upper bound
        # Index 2: friction
        self.bounds_params[2, 1] = 0.5         # friction upper bound
        # Index 3: delay
        self.bounds_params[3, 1] = 10.0        # delay upper bound


@configclass
class Y1_1PaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Y1_1 robot in Pace Sim2Real environment.
    """

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=os.path.join(
                project_root(),
                "source",
                "Y1_1Pace",
                "Y1_1Pace",
                "tasks",
                "manager_based",
                # "Y1_1_robot",
                # "urdf",
                # "Y1_1_Link.urdf"
                "DM8006",
                "urdf",
                "DM8006.urdf"
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
