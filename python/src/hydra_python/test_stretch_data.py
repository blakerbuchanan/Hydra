import pickle, os
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import trange
from typing import NamedTuple


import rerun as rr

from hydra_python import RRLogger
from hydra_python.utils import initialize_hydra_pipeline_stretch
from hydra_python.utils import hydra_get_mesh
from hydra_python.stretch_ai_utils.utils import write_config_yaml

from stretch.core import get_parameters
from stretch.perception import create_semantic_sensor

def main(cfg):
    output_path = cfg.output_path
    os.makedirs(cfg.output_path, exist_ok=True)
    output_path = Path(cfg.output_path)

    rr_logger = RRLogger(output_path)

    with open(cfg.data.test_data_path, 'rb') as file:
        data = pickle.load(file)

    camera_K = data['camera_K'][0]
    # width = data['rgb'][0].shape[1]
    # height = data['rgb'][0].shape[0]
    # camera_info = {
    #     "fx": float(camera_K[0,0]),
    #     "fy": float(camera_K[1,1]),
    #     "cx": float(camera_K[0,2]),
    #     "cy": float(camera_K[1,2]),
    #     "width": width,
    #     "height": height,
    # }

    parameters = get_parameters("/home/saumyas/semnav_workspace/src/hydra/python/src/hydra_python/stretch_ai_utils/cfg/test_stretch.yaml")
    device_id = parameters.data['device_id_sem_sensor']

    Obs = NamedTuple("Obs", [("camera_K", None), ("rgb", None)])
    obs = Obs(camera_K=camera_K, rgb=data['rgb'][0])

    semantic_sensor = create_semantic_sensor(
        parameters=parameters,
        device_id=device_id,
        verbose=True,
    )
    sensor_categories_mapping = semantic_sensor.seg_id_to_name
    # write_config_yaml(sensor_categories_mapping)
    pipeline = initialize_hydra_pipeline_stretch(cfg.hydra, obs, output_path, sensor_categories_mapping)

    for t in trange(len(data['camera_poses'])):
        camera_pose = data['camera_poses'][t]
        rotation_matrix = camera_pose[:3, :3]
        position = camera_pose[:3, 3]
        quat_wxyz = Rotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)
        
        labels = np.ones(data['depth'][t].shape, dtype=np.uint8)

        import ipdb; ipdb.set_trace()
        pipeline.step(t, position, quat_wxyz, data['depth'][t].astype(np.float32), data['semantic'][t], data['rgb'][t])
        
        mesh_vertices, mesh_colors, mesh_triangles = hydra_get_mesh(pipeline)
        rr_logger.log_mesh_data(mesh_vertices, mesh_colors, mesh_triangles)

        rr.log(f"{rr_logger.primary_camera_entity}/rgb", rr.Image(data['rgb'][t]).compress(jpeg_quality=95))
        rr.log(f"{rr_logger.primary_camera_entity}/semantic", rr.SegmentationImage(data['semantic'][t]))
        rr.log(f"{rr_logger.primary_camera_entity}/instance", rr.SegmentationImage(data['instance'][t]))
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