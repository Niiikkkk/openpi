"""Minimal client that queries a pi0 policy server for a bimanual UR7e and prints the predicted actions.

This only depends on `openpi_client` (no JAX/PyTorch needed), so it's meant to run on the robot-side
machine. Fill in the `Ur7eRobot` stubs below with your actual driver calls, then run:

    uv run examples/ur7e/client.py --host <server_host> --port 8000

By default this runs in dry-run mode (prints the predicted actions instead of sending them to the
robot) -- the server config (`pi0_ur7e`) currently skips normalization and isn't fine-tuned on this
robot, so sanity-check a few predictions before passing `--no-dry-run`.
"""

import dataclasses
import time
import sim_state as sim_state

import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy


@dataclasses.dataclass
class Args:
    # Host and port of the policy server (see scripts/serve_policy.py).
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None

    # Natural-language instruction sent to the model on every step.
    prompt: str = "pick the red square"

    # Must match the model's action_horizon for the served config (pi0 defaults to 50).
    action_horizon: int = 50
    # Number of control steps to run.
    max_timesteps: int = 600
    # Sleep between steps, in seconds (matches your control rate).
    control_dt: float = 0.1
    open_loop_horizon: int = 30

    # If true (default), predicted actions are printed instead of sent to the robot.
    dry_run: bool = True


# Indices into the [14] state/action layout that are joint angles (radians, potentially multi-turn
# on the real/sim driver), as opposed to the gripper dims (6, 13).
_JOINT_IDX = [0, 1, 2, 3, 4, 5]

def follow_trajectory():
    import h5py

    hd5f_file = "/home/adminalp/Desktop/IsaacLab/datasets/final_dataset_33.hdf5"
    h5f = h5py.File(hd5f_file, "r")["data"]["demo_0"]

    # Teleport the scene to this episode's recorded initial_state before playback -- mirrors
    # env.reset_to() in IsaacLab's replay_demos.py. Table/cube/cup are real dynamic rigid bodies,
    # not fixed/kinematic, so without this they're left wherever they physically settled since the
    # sim process started, not necessarily where this specific episode's trajectory assumes them
    # to be.
    initial_state = h5f["initial_state"]

    # Re-run SimulationContext.reset() here (not just once at startup in run_environment) --
    # this may clear more physics-internal state (contact caches, warm-start data, etc.) than our
    # own position/velocity-only resets on individual prims do, which is the kind of thing a manual
    # GUI Stop/Play was fixing that nothing else tried so far has replicated. Must run BEFORE the
    # pose overrides below, since it resets everything back to the stage's originally authored
    # state first.
    sim_state.state.sim_context.reset()

    # Sanity-check the articulation's own root Xform against what this episode was actually
    # recorded at -- a raw open_stage() never teleports this the way IsaacLab's per-episode reset
    # does, so any mismatch here offsets the whole arm (and gripper) by a constant amount in world
    # space, which looks exactly like "too high" / "wrong on X" everywhere in the trajectory.
    expected_root_pose = np.asarray(initial_state["articulation"]["robot_1"]["root_pose"])[0]
    actual_root_pos, actual_root_quat = sim_state.state.get_robot_root_pose()
    print("robot root pose -- expected:", expected_root_pose[:3], expected_root_pose[3:],
          "actual (pre-reset):", actual_root_pos, actual_root_quat)
    sim_state.state.reset_robot_root(expected_root_pose[:3], expected_root_pose[3:])

    joint_pos_0 = np.asarray(initial_state["articulation"]["robot_1"]["joint_position"])[0, :7]
    sim_state.state.reset_robot_joints(joint_pos_0, joint_indices=[0, 1, 2, 3, 4, 5, 6])
    for name in ("table", "cube", "cup"):
        root_pose = np.asarray(initial_state["rigid_object"][name]["root_pose"])[0]
        sim_state.state.reset_rigid_object(name, root_pose[:3], root_pose[3:])
    # sim_state.state.tick()

    recorded_eef = np.asarray(h5f["obs"]["eef_pose"])
    recorded_cube_pose = np.asarray(h5f["states"]["rigid_object"]["cube"]["root_pose"])
    recorded_effort = np.asarray(h5f["joint_torque"]["applied"])

    # for step, action in enumerate(h5f["joint_pos_target"][0:]):
    for step, action in enumerate(h5f["joint_pos"][0:]):
        # action[-1] = h5f["joint_pos_target"][step][-1]
        sim_state.state.move_robot(action, sim_state.state.robot_1, indices=[0,1,2,3,4,5,6])
        sim_state.state.tick()
        actual = sim_state.state.robot_1.get_joint_positions()[0, :7]
        if 50 <= step <= 90:
            error = np.asarray(action) - np.asarray(actual)
            efforts = sim_state.state.get_applied_joint_efforts()[0]
            eef_pos, eef_quat = sim_state.state.get_eef_pose()
            cube_pos, cube_quat = sim_state.state.rigid_objects["cube"].get_world_poses()
            cube_pos = cube_pos[0]
            print(step, "effort actual", np.round(np.asarray(efforts), 2),
                  "effort recorded", np.round(recorded_effort[step], 2))
            print(step, "commanded", np.round(action, 3), "actual", np.round(actual, 3), "error", np.round(error, 3),
                  "applied_effort_gripper", np.round(efforts[6], 3))
            print(step, "cube_pos actual", np.round(np.asarray(cube_pos), 4),
                  "cube_pos recorded", np.round(recorded_cube_pose[step, :3], 4))
            print(step, "eef_z actual", round(float(eef_pos[2]), 4), "eef_z recorded", round(float(recorded_eef[step, 2]), 4))
        time.sleep(0.1)
        


def _resolve_multiturn(current_raw: np.ndarray, target_wrapped: np.ndarray) -> np.ndarray:
    """Given the robot's true (possibly multi-turn) current joint angles and a target wrapped to
    (-pi, pi] (the model's single-turn convention), returns the equivalent target closest to
    current_raw. Without this, commanding the raw wrapped value can send the joint the "long way
    around" whenever the wrapped target and the current raw position straddle a +-pi boundary.
    """
    diff = np.mod(target_wrapped - current_raw + np.pi, 2 * np.pi) - np.pi
    return current_raw + diff


class Ur7eRobot:
    """Isaac Sim stand-in for the UR7e driver, backed by sim_state.

    robot_1 -> left arm, robot_2 -> right arm. Note: sim_state's joint positions/gripper
    are in Isaac Sim's native ranges, not the real robot's (native driver radians,
    gripper in (-47, 0)) that `UR7eInputs` normalization expects -- sanity-check predictions
    with --dry-run before trusting `send_action` on real ranges.
    """

    def normalize_gripper(self, gripper: float) -> float:
        return 1.0 + gripper / np.deg2rad(75.0) # radians -> [0,1] 1=open, 0=closed
    
    def unnormalize_gripper(self, gripper: float) -> float:
        degrees = -75.0 * (1.0 - gripper)  # [0,1] -> [-75,0] degrees
        return np.deg2rad(degrees)

    def get_state(self) -> np.ndarray:
        """Returns the current [14] state: [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]."""
        obs = sim_state.state.get_observation()
        left = obs["joint_positions_1"][0]
        left[6] = np.rad2deg(left[6])
        right = obs["joint_positions_2"][0]
        right[6] = np.rad2deg(right[6])
        return np.concatenate([left, right])

    def get_images(self) -> dict[str, np.ndarray]:
        """Returns {"cam_high": ..., "cam_left_wrist": ..., "cam_right_wrist": ...}, each [3, H, W] uint8."""
        obs = sim_state.state.get_observation()
        return {
            "cam_high": obs["top"],
            "cam_left_wrist": obs["wrist_1"],
            "cam_right_wrist": obs["wrist_2"],
        }
    
    def get_observation(self, args: Args) -> dict:
        """Returns the current observation in the same format as `UR7eInputs` expects."""
        obs = sim_state.state.get_observation()
        state = obs["joint_positions_1"][0][:-1]
        gripper = obs["joint_positions_1"][:,-1][0]

        #gripper = self.normalize_gripper(gripper)

        state = np.concatenate([state, [gripper]])

        print(state)

        top_image = obs["top"]
        wrist_image = obs["wrist_1"]
        return {
            "image": top_image,
            "wrist_image": wrist_image,
            "state": state,
            "prompt": args.prompt,
        }    

    def post_process_action(self, action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        action = action.copy()
        # action[:, _JOINT_IDX] = _resolve_multiturn(current_state[_JOINT_IDX], action[:, _JOINT_IDX])
        #action[:,6] = self.unnormalize_gripper(action[:,6])
        # Make to absolute positions
        #action[:,_JOINT_IDX] = action[:,_JOINT_IDX] + current_state[_JOINT_IDX]
        return action
        

    def send_action(self, action: np.ndarray) -> None:
        """Sends a single [14] action (same layout as get_state()) to the robot.

        `current_state` is the robot's true (possibly multi-turn) joint state, used to resolve
        the model's single-turn joint targets to the nearest equivalent angle -- see
        `_resolve_multiturn`.
        """
        sim_state.state.move_robot(action[:7], sim_state.state.robot_1, indices=[0,1,2,3,4,5,6])
        

def main(args: Args) -> None:
    robot = Ur7eRobot()

    base_policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
    print("Server metadata:", base_policy.get_server_metadata())

    actions_from_chunk_completed = 0

    for step in range(args.max_timesteps):
        sim_state.state.tick()

        

        if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
            actions_from_chunk_completed = 0

            obs = robot.get_observation(args)

            start = time.monotonic()
            result = base_policy.infer(obs)
            elapsed_ms = (time.monotonic() - start) * 1000

            actions = np.asarray(result["actions"])

            print(f"[step {step}] state={obs['state']}")

            print(f"[step {step}] PRE_PROCESSING {elapsed_ms:.1f} ms, action={actions}")

            # actions = robot.post_process_action(actions, obs["state"])
        
            
            # print(f"[step {step}] POST_PROCESSING {elapsed_ms:.1f} ms, action={actions}")
        action = actions[actions_from_chunk_completed]
        actions_from_chunk_completed += 1

        if args.dry_run:
            continue
        robot.send_action(action)

        time.sleep(args.control_dt)


# Injected into an already-running Isaac Sim process via connect_to_isaac.py -- there's no
# argv to parse here (this shares sys.argv with whatever launched runner.py), so edit these
# values directly rather than passing CLI flags.
# main(Args(host="localhost", port=8000, prompt="pick up the red cube and place in the blue cup", dry_run=False, control_dt=0.1))

follow_trajectory()

# print(sim_state.state.robot_1._joint_names_to_idx)

# moves = [-0.55661745,  -0.59761634 ,  2.10769024 , -3.14 ,  1.58746856, 0.32365059 ,-37.81349886]

# sim_state.state.move_robot(moves, sim_state.state.robot_1, indices=[0,1,2,3,4,5,6])

