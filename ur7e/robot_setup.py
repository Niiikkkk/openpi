"""Configures a raw isaacsim.core.prims.Articulation to match the actuator gains + USD patches
that IsaacLab's `ImplicitActuatorCfg`/spawner apply to `robot_1` during data collection (see
`ur7e_bimanual_pick/tasks/ur7e_bimanual_pick_env_cfg.py` and `.../tasks/spawners.py`).

Values below are copied from `ur7e_bimanual_pick_env_cfg.py`'s `_ARM_ACTUATOR_ROBOT1_JOINT1..6`
and `_GRIPPER_ACTUATOR` -- if those change, update `_ARM_GAINS`/`_GRIPPER_GAINS` here too, there's
no automatic link between the two files.
"""

import math

import numpy as np
from pxr import Usd

import usd_patches

# (stiffness, damping, effort_limit_N_m, velocity_limit_rad_s) per arm joint, in _ARM_JOINTS order.
_ARM_GAINS = [
    (1230.6, 134.1, 150.0, math.radians(120.0)),  # arm1_shoulder_joint
    (2100.0, 113.9, 150.0, math.radians(220.0)),  # arm1_upperarm_joint
    (1550.0, 90.4, 150.0, math.radians(180.0)),  # arm1_forearm_joint
    (381.8, 17.3, 56.0, math.radians(180.0)),  # arm1_wrist1_joint
    (396.0, 11.0, 56.0, math.radians(180.0)),  # arm1_wrist2_joint
    (396.5, 5.3, 56.0, math.radians(180.0)),  # arm1_wrist3_joint
]
# finger_joint: stiffness, damping, effort_limit -- no separate velocity_limit_sim set in env_cfg.
_GRIPPER_GAINS = (11.5, 0.2, 5.0)


def patch_stage_before_initialize(robot_prim: Usd.Prim) -> None:
    """Call this on the robot's root prim right after `open_stage()`, before `Articulation(...)`
    is constructed/initialized -- the solver-iteration-count and mimic-joint patches are USD schema
    edits that should be in place before PhysX parses/cooks the articulation.
    """
    usd_patches.patch_robot_prim(robot_prim)


def configure_gains_after_initialize(robot) -> None:
    """Call this once after `robot.initialize()`. `robot` is an
    `isaacsim.core.prims.Articulation` (or `ArticulationView`) instance.

    NOTE: `set_gains`/`set_max_efforts`/`set_max_joint_velocities` are the standard method names on
    Isaac Sim's core Articulation API across recent versions, but this hasn't been run against a
    live session -- if any of these raise AttributeError, run
    `print([m for m in dir(robot) if not m.startswith('_')])` in the live session and tell me the
    actual method names so this can be corrected.
    """
    joint_indices = list(range(7))  # 6 arm joints + finger_joint, matching _ARM_JOINTS + gripper order
    kps = np.array([g[0] for g in _ARM_GAINS] + [_GRIPPER_GAINS[0]])
    kds = np.array([g[1] for g in _ARM_GAINS] + [_GRIPPER_GAINS[1]])
    max_efforts = np.array([g[2] for g in _ARM_GAINS] + [_GRIPPER_GAINS[2]])
    max_velocities = np.array([g[3] for g in _ARM_GAINS] + [np.inf])  # no velocity_limit_sim on the gripper

    robot.set_gains(kps=kps, kds=kds, joint_indices=joint_indices)
    robot.set_max_efforts(max_efforts, joint_indices=joint_indices)
    robot.set_max_joint_velocities(max_velocities, joint_indices=joint_indices)
