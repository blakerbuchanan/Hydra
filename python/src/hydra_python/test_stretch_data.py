import pickle
from hydra_python import RRLogger
from pathlib import Path
import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import trange
if __name__=="__main__":
    output_path = Path("/home/saumyas/catkin_ws_semnav/src/hydra/outputs/random")
    rr_logger = RRLogger(output_path)

    input_path = "/home/saumyas/Downloads/stretch_data/stretch_output_2024-10-29_19-55-59.pkl"
    with open(input_path, 'rb') as file:
        data = pickle.load(file)

    # print(data.keys())
    # for k, v in data.items():
    #     if v is None:
    #         print(f"{k} is none")
    #     print(f'Len of key {k} {len(v)}')

    for t in trange(len(data['camera_poses'])):
        rr.log(f"{rr_logger.primary_camera_entity}/rgb", rr.Image(data['rgb'][t]).compress(jpeg_quality=95))
        rr.log(f"{rr_logger.primary_camera_entity}/semantic", rr.DepthImage(data['depth'][t], meter=1.0))
        camera_pose = data['camera_poses'][t]
        rotation_matrix = camera_pose[:3, :3]
        position = camera_pose[:3, 3]
        quat_wxyz = Rotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)
        rr_logger.log_camera_tf(position, quat_wxyz)
        rr_logger.step()

