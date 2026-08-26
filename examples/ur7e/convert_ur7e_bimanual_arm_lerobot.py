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

REPO_NAME = "ur7e_bimanual_arm"  # Name of the output dataset, also used for the Hugging Face Hub


def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = Path("./datasets") / REPO_NAME
    print(f"Output path: {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)
    data_dir = Path(data_dir)

    # Create LeRobot dataset, define features to store
    # We will follow the DROID data naming conventions here.
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        root=output_path,
        repo_id=REPO_NAME,
        robot_type="ur7e_bimanual",
        fps=20,  
        features={
            "image": {
                "dtype": "image",
                "shape": (320, 480, 3),  # This is the resolution used in the DROID RLDS dataset
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (320, 480, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_right": {
                "dtype": "image",
                "shape": (320, 480, 3),
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
            
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    
    episode_paths = list(data_dir.glob("dataset_cable_generated_filtered.hdf5"))
    print(f"Found {len(episode_paths)} episodes for conversion")

    # We will loop over each dataset_name and write episodes to the LeRobot dataset
    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        demos = h5py.File(episode_path, "r")["data"]
        for demo in tqdm(demos.items(), desc="Converting demos"):
            name, demo_data = demo
            steps = demo_data["actions"].shape[0]
            for step in range(steps):
                dataset.add_frame(
                    {
                        "image": demo_data["obs"]["top_camera"][step],
                        "wrist_image_left": demo_data["obs"]["wrist_cam_1"][step],
                        "wrist_image_right": demo_data["obs"]["wrist_cam_2"][step],
                        "state": np.concatenate(([demo_data["obs"]["joint_pos_left"][step]], [demo_data["obs"]["joint_pos_right"][step]]), axis=1).squeeze(),
                        "actions": np.concatenate(([demo_data["obs"]["joint_pos_target_left"][step]], [demo_data["obs"]["joint_pos_target_right"][step]]), axis=1).squeeze(),
                        "task": "Pick up the cable and lift it up",
                    
                    }
                )
            dataset.save_episode()

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