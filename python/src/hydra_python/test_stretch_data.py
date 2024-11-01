import pickle, os
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import trange

import rerun as rr

from hydra_python import RRLogger
from hydra_python.utils import initialize_hydra_pipeline_rosbag
from hydra_python.run import hydra_get_mesh

def main(cfg):
    output_path = cfg.output_path
    os.makedirs(cfg.output_path, exist_ok=True)
    output_path = Path(cfg.output_path)

    rr_logger = RRLogger(output_path)

    with open(cfg.data.test_data_path, 'rb') as file:
        data = pickle.load(file)

    camera_K = data['camera_K'][0]
    width = data['rgb'][0].shape[1]
    height = data['rgb'][0].shape[0]
    camera_info = {
        "fx": float(camera_K[0,0]),
        "fy": float(camera_K[1,1]),
        "cx": float(camera_K[0,2]),
        "cy": float(camera_K[1,2]),
        "width": width,
        "height": height,
    }

    pipeline = initialize_hydra_pipeline_rosbag(cfg.hydra, camera_info, output_path)

    for t in trange(len(data['camera_poses'])):
        camera_pose = data['camera_poses'][t]
        rotation_matrix = camera_pose[:3, :3]
        position = camera_pose[:3, 3]
        quat_wxyz = Rotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)
        
        labels = np.ones(data['depth'][t].shape, dtype=np.uint8)

        pipeline.step(t, position, quat_wxyz, data['depth'][t].astype(np.float32), labels, data['rgb'][t])
        mesh_vertices, mesh_colors, mesh_triangles = hydra_get_mesh(pipeline)
        rr_logger.log_mesh_data(mesh_vertices, mesh_colors, mesh_triangles)

        rr.log(f"{rr_logger.primary_camera_entity}/rgb", rr.Image(data['rgb'][t]).compress(jpeg_quality=95))
        rr.log(f"{rr_logger.primary_camera_entity}/depth", rr.DepthImage(data['depth'][t], meter=1.0))
        rr_logger.log_camera_tf(position, quat_wxyz)
        rr_logger.step()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / 'commands' / 'cfg' / f'{args.cfg_file}.yaml'
    cfg = OmegaConf.load(config_path)

    OmegaConf.resolve(cfg)
    main(cfg)