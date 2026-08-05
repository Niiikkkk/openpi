"""
Shared, persistent simulation state. Imported both by runner.py (which
calls setup() once, after the stage is loaded and playing) and by code
injected via the python_server socket (test2.py), which only reads from
it. Cameras/robots are created exactly once per process — never rebuild
them per injected call, that's what leaked VRAM before.
"""

import numpy as np

from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import Articulation, RigidPrim, XFormPrim
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.sensors.experimental.rtx import RtxCamera
from isaacsim.sensors.experimental.rtx import CameraSensor

import robot_setup

CAMERA_RESOLUTION = (320, 480)
#WRIST_CAMERA_RESOLUTION = (320,180)
# Matches ur7e_bimanual_pick_env_cfg.py's self.sim.dt / self.decimation (100Hz physics, 20Hz
# control) -- keep these in sync with that file if it changes. A raw open_stage()+timeline.play()
# runs at whatever timestep the USD/app happens to fall back to, with no fixed relationship to
# real simulated time -- confirmed empirically to cause an inconsistent "closes too slowly, like
# something is resisting it" symptom that changed depending on GUI Stop/Play state, i.e. a timing
# artifact rather than a real physical force.
_PHYSICS_DT = 0.01
_DECIMATION = 5


class SimState:
    def __init__(self):
        self.robot_1 = None
        self.cameras = {}
        self.rigid_objects = {}
        self.eef = None
        self._initialized = False
        self.simulation_app = None
        self.sim_context = None

    def _make_camera(self, prim_path):
        rt_cam = RtxCamera(prim_path)
        cam = CameraSensor(rt_cam, resolution=CAMERA_RESOLUTION)
        cam._initialize_sensor(annotators=["rgb"])
        return cam

    def run_environment(self, usd_path, args, sim_app, robot_1_path=None):
        self.simulation_app = sim_app

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.code_editor.python_server")

        from isaacsim.core.experimental.utils.stage import is_stage_loading, open_stage

        open_stage(usd_path)
        while is_stage_loading():
            self.simulation_app.update()

        # Apply the USD-level patches (dedupe articulation roots, raised solver iterations,
        # gripper mimic joints) BEFORE the SimulationContext is constructed/reset -- PhysX cooks
        # the whole stage's physics schemas once when the sim actually starts (`sim_context.reset()`
        # below), independent of when `Articulation.initialize()` runs later in `setup()`. Applying
        # these patches AFTER that first cook (as `setup()` used to) means they're authored onto the
        # USD but never picked up by the already-cooked simulation -- confirmed empirically: a
        # manual GUI Stop/Play (which forces PhysX to fully re-cook from the current, patched USD)
        # visibly fixed gripper behavior that persisted broken across every other change tried,
        # including this file's own timestep-determinism fix. Without the mimic joints specifically,
        # the gripper's passive knuckle/finger linkage joints have no constraint tying them to
        # finger_joint, so contact force at the fingertip just freely rotates them instead of
        # holding the 4-bar linkage rigid. The solver-iteration bump goes with it since a stiff
        # mimic constraint needs enough iterations to actually converge, or it looks unstable/
        # under-converged the same way insufficient stiffness would.
        if robot_1_path is not None:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            robot_prim = stage.GetPrimAtPath(robot_1_path)
            robot_setup.patch_stage_before_initialize(robot_prim)

        # Explicit physics_dt, instead of timeline.play() + simulation_app.update() at whatever
        # rate the USD/app happens to run -- see _PHYSICS_DT's comment above for why this matters.
        self.sim_context = SimulationContext(physics_dt=_PHYSICS_DT, rendering_dt=_PHYSICS_DT * _DECIMATION)
        self.sim_context.reset()

        self.simulation_app.update()

    def setup(self, robot_1_path, camera_paths, rigid_object_paths=None, eef_path=None):
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

        # USD-level patches already applied in `run_environment`, before the sim started -- see its
        # docstring/comment for why that ordering matters.
        self.robot_1 = Articulation(prim_paths_expr=robot_1_path)
        self.robot_1.initialize()
        robot_setup.configure_gains_after_initialize(self.robot_1)

        self.cameras = {name: self._make_camera(path) for name, path in camera_paths.items()}

        if rigid_object_paths:
            self.rigid_objects = {name: RigidPrim(prim_paths_expr=path) for name, path in rigid_object_paths.items()}
            for obj in self.rigid_objects.values():
                obj.initialize()

        if eef_path:
            self.eef = XFormPrim(prim_paths_expr=eef_path)
            self.eef.initialize()

        self._initialized = True

    def get_eef_pose(self):
        """Returns (pos[3], quat_wxyz[4]) of the end-effector link's world Xform -- for comparing
        against the recorded `obs/eef_pose` (xyz + quat) in the hdf5 to see how far off the actual
        replayed end-effector position is, independent of which joint-space convention/offset is
        involved.
        """
        positions, orientations = self.eef.get_world_poses()
        return positions[0], orientations[0]

    def reset_rigid_object(self, name, pos, quat_xyzw):
        """Teleport a dynamic rigid object (table/cube/cup) to `pos`/`quat_xyzw` and zero its
        velocity. `quat_xyzw` is the (x, y, z, w) order the hdf5's
        `initial_state/rigid_object/<name>/root_pose` is actually stored in (confirmed empirically:
        the robot root's recorded (0, 0, 0, 1) is identity, which is only true read as (x, y, z, w)
        -- read as this core API's own (w, x, y, z) convention it's a 180 degree yaw). Converted to
        (w, x, y, z) here before handing off to `RigidPrim.set_world_poses`.
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
        `quat_xyzw` is (x, y, z, w) -- see `reset_rigid_object`'s docstring for why.
        """
        w, x, y, z = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
        self.robot_1.set_world_poses(positions=np.asarray([pos]), orientations=np.asarray([[w, x, y, z]]))
        self.robot_1.set_velocities(np.zeros((1, 6)))

    def reset_robot_joints(self, joint_positions, joint_indices=None):
        """Teleport the robot's joints (position + zero velocity) and re-point the position-target
        buffer at the same values -- without re-targeting, the PD drive would immediately fight to
        pull the joints back to whatever target was last commanded before this teleport.
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
            for i in range(_DECIMATION):
                self.sim_context.step(render=(i == _DECIMATION - 1))

    def get_applied_joint_efforts(self):
        """Returns the actually-applied (post effort-limit-clip) joint torques/forces for indices
        [:7], to check whether a joint is saturated at its effort limit -- e.g. finger_joint
        (index 6) pinned near its `effort_limit_sim` would mean its rest position is decided by
        external contact resistance, not by stiffness/damping, the same way training's own
        recorded `joint_torque/applied` shows it saturated at -1.0 throughout the hold phase.
        Tries a couple of known isaacsim.core.prims.Articulation method names since this hasn't
        been checked against a live session -- if both raise AttributeError, run
        `print([m for m in dir(sim_state.state.robot_1) if not m.startswith('_')])` in the live
        session and report the actual method name back.
        """
        for method_name in ("get_measured_joint_efforts", "get_applied_joint_efforts"):
            method = getattr(self.robot_1, method_name, None)
            if method is not None:
                return method()[:, :7]
        raise AttributeError(
            "Neither get_measured_joint_efforts nor get_applied_joint_efforts exists on this "
            "Articulation -- run print([m for m in dir(sim_state.state.robot_1) if not "
            "m.startswith('_')]) and report the actual method name."
        )

    def get_rgb(self, name):
        return self.cameras[name].get_data("rgb")[0].numpy()

    def get_observation(self):
        return {
            "top": self.get_rgb("top"),
            "wrist_1": self.get_rgb("wrist_1"),
            "joint_positions_1": self.robot_1.get_joint_positions()[:,:7],
            "joint_velocities_1": self.robot_1.get_joint_velocities()[:,:7],
        }

    def move_robot(self, action, robot, indices=None):
        robot.apply_action(ArticulationActions(joint_positions=action,joint_indices=indices))


# Singleton — created once per process, shared by runner.py and injected code.
state = SimState()
