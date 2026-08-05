"""
Persistent Isaac Sim process: launch once, keep it alive, and drive the
app update loop yourself. `test.py` keeps working unchanged — it connects
to the same python_server socket (127.0.0.1:8226) and injects code into
this already-running process instead of a GUI instance.

Same raw-stage approach as runner_single.py, but uses sim_state_patched (not sim_state) --
see robot_setup.py/usd_patches.py for what that adds (matching actuator gains + USD patches
to the environment that actually recorded the training dataset).

Usage:
    python runner_single_patched.py            # headless
    python runner_single_patched.py --gui      # with viewport window
"""

import argparse

ROOT = "/World"
ROBOT_1 = ROOT + "/ur7e_1"
WRIST_CAMERA_1_PATH = ROBOT_1 + "/Geometry/arm1_base_link/arm1_shoulder_link/arm1_upperarm_link/arm1_forearm_link/arm1_wrist1_link/arm1_wrist2_link/arm1_wrist3_link/arm1_end_effector_link/daA2500_14uc_1"  # placeholder
TOP_CAMERA_PATH = ROOT + "/VCXG_2_51C"
# Source-USD child names, verbatim (see ur7e_bimanual_pick_env_cfg.py's SubPrimReferenceCfg
# comments) -- the dynamic (non-kinematic) rigid bodies a replayed episode needs teleported back to
# their recorded initial_state pose before playback.
RIGID_OBJECT_PATHS = {"table": ROOT + "/Table", "cube": ROOT + "/Cube", "cup": ROOT + "/container"}

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true", help="show the Isaac Sim GUI window")
args = parser.parse_args()


from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": not args.gui})


import sim_state_patched as sim_state

sim_state.state.run_environment("/home/adminalp/Desktop/Table_with_robots_single_robot.usd", args, simulation_app)

sim_state.state.setup(ROBOT_1, {"top": TOP_CAMERA_PATH, "wrist_1": WRIST_CAMERA_1_PATH}, rigid_object_paths=RIGID_OBJECT_PATHS)
sim_state.state.tick()

print("Simulation is running...")

print("runner ready — python_server listening on 127.0.0.1:8226, ticking now")

try:
    while simulation_app.is_running():
        simulation_app.update()
except KeyboardInterrupt:
    pass
finally:
    print("Stopping simulation...")
    simulation_app.close()
