"""
Shared, persistent simulation state. Imported both by runner.py (which
calls setup() once, after the stage is loaded and playing) and by code
injected via the python_server socket (test2.py), which only reads from
it. Cameras/robots are created exactly once per process — never rebuild
them per injected call, that's what leaked VRAM before.
"""

import numpy as np

from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.sensors.experimental.rtx import RtxCamera
from isaacsim.sensors.experimental.rtx import CameraSensor

CAMERA_RESOLUTION = (256, 256)
#WRIST_CAMERA_RESOLUTION = (320,180)
N_TICKS = 10  # ticks to wait for annotator data to become valid


class SimState:
    def __init__(self):
        self.robot_1 = None
        self.cameras = {}
        self._initialized = False
        self.simulation_app = None

    def _make_camera(self, prim_path):
        rt_cam = RtxCamera(prim_path)
        cam = CameraSensor(rt_cam, resolution=CAMERA_RESOLUTION)
        cam._initialize_sensor(annotators=["rgb"])
        return cam

    def run_environment(self, usd_path, args, sim_app):
        self.simulation_app = sim_app

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.code_editor.python_server")

        import omni.timeline as tl

        from isaacsim.core.experimental.utils.stage import is_stage_loading, open_stage

        open_stage(usd_path)
        while is_stage_loading():
            self.simulation_app.update()

        timeline = tl.get_timeline_interface()
        timeline.play()

        self.simulation_app.update()

    def setup(self, robot_1_path, camera_paths):
        """camera_paths: dict like {"top": "...", "wrist_1": "...", "wrist_2": "..."}"""
        if self._initialized:
            return

        self.robot_1 = Articulation(prim_paths_expr=robot_1_path)
        self.robot_1.initialize()

        self.cameras = {name: self._make_camera(path) for name, path in camera_paths.items()}

        self._initialized = True

    def is_ready(self):
        return self._initialized

    def tick(self, n=N_TICKS):
        for _ in range(n):
            self.simulation_app.update()

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
