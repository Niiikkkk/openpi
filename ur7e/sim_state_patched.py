"""
Shared, persistent simulation state -- same raw open_stage()/isaacsim.core.prims.Articulation
approach as sim_state.py, but with robot_setup.py's USD patches (solver iterations, gripper mimic
joints) and actuator gains (stiffness/damping/effort/velocity limits) applied, so this robot
matches the one that actually recorded the training dataset instead of running PhysX's stock
defaults. See robot_setup.py's docstring for exactly what's being replicated and why.

Also fixes a second gap: `ur7e_bimanual_pick_env_cfg.py` sets `sim.dt=0.01` and `decimation=5`
(20Hz control) as IsaacLab *Python-level* SimulationCfg settings -- those are never baked into the
USD file itself, so a raw `open_stage()` + `timeline.play()` runs at whatever the USD's own
authored PhysicsScene timestep (or Isaac Sim's own default) happens to be, and `tick()`'s old
`simulation_app.update()`-based stepping had no fixed relationship to real simulated time at all.
Using `isaacsim.core.api.SimulationContext` with an explicit `physics_dt`/decimation loop instead
makes one `tick()` call advance exactly one 20Hz control step, matching the rate the dataset was
recorded at -- without this, recorded joint targets (including the gripper's) play back at the
wrong effective speed relative to real dynamics, which looks exactly like "closes too soon" even
with correct gains/mimic joints.

Imported both by runner.py (which calls setup() once, after the stage is loaded and playing) and
by code injected via the python_server socket (test2.py), which only reads from it. Cameras/robots
are created exactly once per process -- never rebuild them per injected call, that's what leaked
VRAM before.
"""

import numpy as np

from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.sensors.experimental.rtx import RtxCamera
from isaacsim.sensors.experimental.rtx import CameraSensor

import robot_setup

CAMERA_RESOLUTION = (320, 480)
# Matches ur7e_bimanual_pick_env_cfg.py's self.sim.dt / self.decimation (100Hz physics, 20Hz
# control) -- keep these in sync with that file if it changes.
_PHYSICS_DT = 0.01
_DECIMATION = 5


class SimState:
    def __init__(self):
        self.robot_1 = None
        self.cameras = {}
        self.rigid_objects = {}
        self._initialized = False
        self.simulation_app = None
        self.sim_context = None

    def _make_camera(self, prim_path):
        rt_cam = RtxCamera(prim_path)
        cam = CameraSensor(rt_cam, resolution=CAMERA_RESOLUTION)
        cam._initialize_sensor(annotators=["rgb"])
        return cam

    def run_environment(self, usd_path, args, sim_app):
        self.simulation_app = sim_app

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.code_editor.python_server")

        from isaacsim.core.experimental.utils.stage import is_stage_loading, open_stage

        open_stage(usd_path)
        while is_stage_loading():
            self.simulation_app.update()

        import omni.timeline as tl

        timeline = tl.get_timeline_interface()
        timeline.play() 

        self.simulation_app.update()

    def setup(self, robot_1_path, camera_paths, rigid_object_paths=None):
        """camera_paths: dict like {"top": "...", "wrist_1": "...", "wrist_2": "..."}

        rigid_object_paths: dict like {"table": "/World/Table", "cube": "/World/Cube",
        "cup": "/World/container"} -- dynamic (non-articulated) rigid bodies in the scene that a
        replayed episode needs teleported back to its recorded `initial_state` before playback,
        the same way `env.reset_to()` does in IsaacLab's `replay_demos.py`. Without this, these
        objects are left wherever they physically settled since the sim process started (they are
        real dynamic RigidBodyAPI prims, not fixed/kinematic -- table included, see
        `ur7e_bimanual_pick_env_cfg.py`'s own comment on that), which does not necessarily match
        the specific episode's recorded object placement.
        """
        if self._initialized:
            return

        # Apply the USD-level patches (solver iterations, gripper mimic joints) BEFORE
        # Articulation.initialize() parses/cooks the articulation -- see robot_setup.py.
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(robot_1_path)
        robot_setup.patch_stage_before_initialize(robot_prim)

        self.robot_1 = Articulation(prim_paths_expr=robot_1_path)
        self.robot_1.initialize()
        robot_setup.configure_gains_after_initialize(self.robot_1)

        self.cameras = {name: self._make_camera(path) for name, path in camera_paths.items()}

        if rigid_object_paths:
            self.rigid_objects = {name: RigidPrim(prim_paths_expr=path) for name, path in rigid_object_paths.items()}
            for obj in self.rigid_objects.values():
                obj.initialize()

        self._initialized = True

    def reset_rigid_object(self, name, pos, quat_xyzw):
        """Teleport a dynamic rigid object (table/cube/cup) to `pos`/`quat_xyzw` and zero its
        velocity. `quat_xyzw` is the (x, y, z, w) order the hdf5's
        `initial_state/rigid_object/<name>/root_pose` is actually stored in (confirmed empirically:
        the robot root's recorded (0, 0, 0, 1) is identity, which is only true read as (x, y, z, w)
        -- read as this core API's own (w, x, y, z) convention it's a 180 degree yaw, which is
        exactly the flip seen before this conversion was added). Converted to (w, x, y, z) here
        before handing off to `RigidPrim.set_world_poses`.
        """
        w, x, y, z = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
        obj = self.rigid_objects[name]
        obj.set_world_poses(positions=np.asarray([pos]), orientations=np.asarray([[w, x, y, z]]))
        obj.set_velocities(np.zeros((1, 6)))

    def get_robot_root_pose(self):
        """Returns (pos[3], quat_wxyz[4]) of the articulation's own root Xform, as currently
        authored/loaded -- compare this against `ArticulationCfg.InitialStateCfg` in
        `ur7e_bimanual_pick_env_cfg.py` (pos=(-0.7261, 0, 1.1022), identity rotation). IsaacLab
        teleports the root to that pose every reset via its own event term; a raw `open_stage()`
        never does, so this is only guaranteed to match if the USD's authored transform truly is
        that exact value -- worth confirming directly rather than assuming, since any mismatch here
        offsets the entire arm (and therefore the gripper) by a constant amount in world space.
        """
        positions, orientations = self.robot_1.get_world_poses()
        return positions[0], orientations[0]

    def reset_robot_root(self, pos, quat_xyzw):
        """Teleport the articulation's own root Xform to `pos`/`quat_xyzw` and zero its velocity.
        `quat_xyzw` is (x, y, z, w) -- see `reset_rigid_object`'s docstring for why (confirmed
        empirically: passing the recorded root_pose's (0, 0, 0, 1) straight through as (w, x, y, z)
        produced a 180 degree yaw instead of the identity rotation it actually is).
        """
        w, x, y, z = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
        self.robot_1.set_world_poses(positions=np.asarray([pos]), orientations=np.asarray([[w, x, y, z]]))
        self.robot_1.set_velocities(np.zeros((1, 6)))

    def reset_robot_joints(self, joint_positions, joint_indices=None):
        """Teleport the robot's joints (position + zero velocity) and re-point the position-target
        buffer at the same values, mirroring `HeldTargetDifferentialInverseKinematicsAction.reset`/
        `ResetSyncedBinaryJointPositionAction.reset` -- without re-targeting, the PD drive would
        immediately fight to pull the joints back to whatever target was last commanded before this
        teleport.
        """
        positions = np.asarray([joint_positions])
        self.robot_1.set_joint_positions(positions, joint_indices=joint_indices)
        self.robot_1.set_joint_velocities(np.zeros_like(positions), joint_indices=joint_indices)
        self.move_robot(joint_positions, self.robot_1, indices=joint_indices)

    def is_ready(self):
        return self._initialized

    def tick(self, n=1):
        """Advance the sim by `n` control steps (each `_DECIMATION` physics substeps of
        `_PHYSICS_DT`), matching the 20Hz rate the dataset was recorded at -- NOT `n` raw
        simulation_app updates like the old N_TICKS-based version.
        """
        for _ in range(n):
            self.simulation_app.update()

    def get_rgb(self, name):
        return self.cameras[name].get_data("rgb")[0].numpy()

    def get_observation(self):
        return {
            "top": self.get_rgb("top"),
            "wrist_1": self.get_rgb("wrist_1"),
            "joint_positions_1": self.robot_1.get_joint_positions()[:, :7],
            "joint_velocities_1": self.robot_1.get_joint_velocities()[:, :7],
        }

    def move_robot(self, action, robot, indices=None):
        robot.apply_action(ArticulationActions(joint_positions=action, joint_indices=indices))


# Singleton -- created once per process, shared by runner.py and injected code.
state = SimState()
