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

REPO_NAME = "ur7e_single_arm"  # Name of the output dataset, also used for the Hugging Face Hub


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
        robot_type="ur7e",
        fps=20,  
        features={
            "image": {
                "dtype": "image",
                "shape": (320, 480, 3),  # This is the resolution used in the DROID RLDS dataset
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (320, 480, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # We will use joint *position* actions here (6D) + gripper position (1D)
                "names": ["actions"],
            },
            
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    
    episode_paths = list(data_dir.glob("final_dataset.hdf5"))
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
                        "wrist_image": demo_data["obs"]["wrist_cam_1"][step],
                        "state": demo_data["obs"]["joint_pos"][step],
                        "actions": demo_data["joint_pos_target"][step],
                        "task": "Pick up the red cube and place it in the blue container",
                    
                    }
                )
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "ur7e", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)