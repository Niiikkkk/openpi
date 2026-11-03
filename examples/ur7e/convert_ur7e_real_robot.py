"""
Minimal example script for converting a dataset collected on the DROID platform to LeRobot format.

Usage:
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

from collections import defaultdict
import copy
import glob
import json
from pathlib import Path
import shutil
import h5py

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

REPO_NAME = "ur7e_real_robot"  # Name of the output dataset, also used for the Hugging Face Hub


def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def main(data_dir: str, *, push_to_hub: bool = False):
    output_path = Path("/media/adminalp/data/lerobot_db") / REPO_NAME
    print(f"Output path: {output_path}")
    data_dir = Path(data_dir)

    # We will follow the DROID data naming conventions here.
    # LeRobot assumes that dtype of image data is `image`
    features = {
        "top_image": {
            "dtype": "image",
            "shape": (720, 1280, 3),  # This is the resolution used in the DROID RLDS dataset
            "names": ["height", "width", "channel"],
        },
        "wrist_image_left": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image_right": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (14,),  # We will use joint *position* actions here (12D) + gripper position (2D)
            "names": ["actions"],
        },
    }

    if output_path.exists():
        # Resume: load the dataset already on disk instead of wiping it, and only convert the
        # episodes that aren't in it yet. The regular constructor (unlike `.create`) doesn't spin
        # up the async image writer itself, so we start it manually with the same settings.
        print(f"Found existing dataset at {output_path}, resuming")
        dataset = LeRobotDataset(repo_id=REPO_NAME, root=output_path)
        dataset.start_image_writer(num_processes=5, num_threads=10)
    else:
        # Create LeRobot dataset, define features to store
        dataset = LeRobotDataset.create(
            root=output_path,
            repo_id=REPO_NAME,
            robot_type="ur7e_bimanual",
            fps=30,
            features=features,
            image_writer_threads=10,
            image_writer_processes=5,
        )

    # Sorted so that "skip the first N episodes" below means the same N episodes on every run -
    # directory listing order is not guaranteed stable across runs/filesystems.
    episode_paths = sorted(data_dir.glob("*.h5"))
    print(f"Found {len(episode_paths)} episodes for conversion")

    already_done = dataset.meta.total_episodes
    if already_done:
        print(f"{already_done} episodes already in the dataset, skipping them")
        episode_paths = episode_paths[already_done:]
    # episode_paths = episode_paths[:1]

    # We will loop over each dataset_name and write episodes to the LeRobot dataset
    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        original_file = str(episode_path).replace("_aligned_v2","")
        original_file = original_file.replace("dataset_aligned_cleaned","")
        with h5py.File(original_file, "r") as original_demo, h5py.File(episode_path, "r") as demo:
            demo_data = demo
            steps = demo_data["images"]["left_camera"]["index"].shape[0]
            for step in range(steps):
                left_joint_pos = demo_data["robot"]["left"]["actual_q"][step]
                left_gripper = [demo_data["robot"]["left"]["gripper_position"][step]]
                rigth_joint_pose = demo_data["robot"]["right"]["actual_q"][step]
                right_gripper = [demo_data["robot"]["right"]["gripper_position"][step]]
                state = np.concatenate((left_joint_pos, left_gripper, rigth_joint_pose, right_gripper), axis=0, dtype=np.float32)
                actions_left = demo_data["robot"]["left"]["commanded_q"][step]
                action_gripper_left = [demo_data["robot"]["left"]["gripper_target_position"][step]]
                actions_right = demo_data["robot"]["right"]["commanded_q"][step]
                action_gripper_right = [demo_data["robot"]["right"]["gripper_target_position"][step]]
                action = np.concatenate((actions_left, action_gripper_left, actions_right, action_gripper_right), axis=0, dtype=np.float32)
                left_cam_idx = demo_data["images"]["left_camera"]["index"][step]
                right_cam_idx = demo_data["images"]["right_camera"]["index"][step]
                top_cam_idx = demo_data["images"]["top_camera"]["index"][step]

                dataset.add_frame(
                    {
                        "top_image": cv2.imdecode(np.asarray(original_demo["images"]["top_camera"]["data"][top_cam_idx], dtype=np.uint8), cv2.IMREAD_COLOR),
                        "wrist_image_left": cv2.imdecode(np.asarray(original_demo["images"]["left_camera"]["data"][left_cam_idx], dtype=np.uint8), cv2.IMREAD_COLOR),
                        "wrist_image_right": cv2.imdecode(np.asarray(original_demo["images"]["right_camera"]["data"][right_cam_idx], dtype=np.uint8), cv2.IMREAD_COLOR),
                        "state": state,
                        "actions": action,
                        "task": "Pick up the Ethernet cable and plug it into the socket",
                    }
                )
            dataset.save_episode()
            # `save_episode` appends this episode's (already-written-to-disk) data onto
            # `dataset.hf_dataset`, an in-memory copy that otherwise grows without bound across
            # episodes (and includes full-resolution images) - OOMs partway through a large
            # conversion. We never read `dataset.hf_dataset` ourselves, so just drop it each time.
            dataset.hf_dataset = dataset.create_hf_dataset()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "ur7e_bimanual", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)