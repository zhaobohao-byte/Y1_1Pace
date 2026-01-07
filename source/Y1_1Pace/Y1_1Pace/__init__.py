# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks.manager_based.pace.pace_sim2real_env_cfg import (
    PaceSim2realEnvCfg,
    PaceSim2realSceneCfg,
    PaceCfg,
)

# Optimizer
from .optim import CMAESOptimizer

__all__ = [
    "PaceSim2realEnvCfg",
    "PaceSim2realSceneCfg",
    "PaceCfg",
    "CMAESOptimizer"
]

