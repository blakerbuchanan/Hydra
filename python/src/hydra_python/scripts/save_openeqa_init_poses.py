# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
from pathlib import Path
from typing import List
import json

import numpy as np
import tqdm
from PIL import Image

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hm3d-root",
        type=Path,
        default="data/scene_datasets/hm3d/val",
        help="path to hm3d scene data (default: data/scene_datasets/hm3d/val)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="/home/saumyas/catkin_ws_semnav/data/openEQA/data/frames/hm3d-v0",
        help="output path (default: data/frames/hm3d-v0)",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="only extract rgb frames (default: false)",
    )
    args = parser.parse_args()
    return args


def extract_frames(output_directory):
    folders = sorted(output_directory.glob("*"))

    idx = 0
    init_poses = {}
    for folder in tqdm.tqdm(folders):
        files = sorted(folder.glob("*.pkl"))
        if len(files) > 0:
            data = pickle.load(files[0].open("rb"))
            quat = data['agent_state'].rotation
            w, x, y, z = quat.w, quat.x, quat.y, quat.z

            roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
            pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))  # Clip to handle numerical issues
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

            # print(f"Roll:{roll}, Pitch:{pitch}, Yaw:{yaw}")
            question_id = str(folder).split('frames/')[1]
            init_poses[question_id] = {
                'idx': idx,
                'scene_id': data['scene_id'],
                'init_pos': [float(x) for x in data['agent_state'].position],
                'quat_wxyz': [quat.w, quat.x, quat.y, quat.z],
                'init_angle': float(pitch),
            }
            idx += 1

    print(f"Saving file: {output_directory}")
    with open(output_directory / "openeqa_init_poses.json", 'w') as file:
        json.dump(init_poses, file, indent=4)

def main(args):
    extract_frames(args.output_directory)

if __name__ == "__main__":
    main(parse_args())