import pickle, os
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import trange
from typing import NamedTuple
from PIL import Image
import rerun as rr

import hydra_python as hydra
from hydra_python import RRLogger
from hydra_python.utils import initialize_hydra_pipeline_stretch
from hydra_python.utils import hydra_get_mesh
from hydra_python.stretch_ai_utils.utils import write_config_yaml

from stretch.core import get_parameters
from stretch.perception import create_semantic_sensor

import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

@dataclass
class Obs:
    camera_K: any = None
    rgb: any = None
    depth: any = None
    semantic: any = None
    instance: any = None
    task_observations: any = None

def main(cfg):
    output_path = cfg.output_path
    os.makedirs(cfg.output_path, exist_ok=True)
    output_path = Path(cfg.output_path)

    rr_logger = RRLogger(output_path)

    with open(cfg.data.test_data_path, 'rb') as file:
        data = pickle.load(file)

    camera_K = data['camera_K'][0]
    for img_idx in range(len(data['semantic'])):
        # img = Image.fromarray(data['semantic'][img_idx])
                # Define or use a colormap for the segmentation
        cmap = plt.get_cmap('Paired')  # Or choose any other colormap (e.g., 'tab20', 'viridis')
        seg_array = data['semantic'][img_idx]

        # Normalize array for colormap
        norm = mcolors.Normalize(vmin=seg_array.min(), vmax=seg_array.max())
        segmentation_rgb = cmap(norm(seg_array))

        # Convert to 8-bit RGB format and save as an image
        segmentation_rgb = (segmentation_rgb[:, :, :3] * 255).astype(np.uint8)  # Discard alpha, scale to 0-255
        seg_img = Image.fromarray(segmentation_rgb)
        seg_img.save(output_path / f"semantic_img_{img_idx}.png")

    import ipdb; ipdb.set_trace()

    for img_idx in range(len(data['rgb'])):
        img = Image.fromarray(data['rgb'][img_idx])
        img.save(output_path / f"rgb_img_{img_idx}.png")

    # final_img = Image.fromarray(np.concatenate([*sampled_images[top_k_indices], imgs_rgb[-1]], axis=1))
    for img_idx in range(len(data['depth'])):
        depth_min, depth_max = data['depth'][img_idx].min(), data['depth'][img_idx].max()
        depth_normalized = 255 * (data['depth'][img_idx] - depth_min) / (depth_max - depth_min)
        depth_normalized = depth_normalized.astype(np.uint8)
        final_img = Image.fromarray(depth_normalized, mode='L')
        final_img.save(output_path / f"depth_img_{img_idx}.png")

    import ipdb; ipdb.set_trace()
    
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
    obs = Obs(camera_K=camera_K, rgb=data['rgb'][0])

    semantic_sensor = create_semantic_sensor(
        parameters=parameters,
        device_id=parameters.data['device_id_sem_sensor'],
        verbose=True,
    )
    sensor_categories_mapping = semantic_sensor.seg_id_to_name
    # write_config_yaml(sensor_categories_mapping)
    # import ipdb; ipdb.set_trace()
    pipeline = initialize_hydra_pipeline_stretch(cfg.hydra, obs, output_path, sensor_categories_mapping)

    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    sg_sim = hydra.SceneGraphSim(
        cfg, 
        output_path, 
        pipeline, 
        rr_logger, 
        device=device)

    for t in trange(len(data['camera_poses'])):
        camera_pose = data['camera_poses'][t]
        rotation_matrix = camera_pose[:3, :3]
        position = camera_pose[:3, 3]
        quat_wxyz = Rotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)
        
        # labels = np.ones(data['depth'][t].shape, dtype=np.uint8)
        # labels = data['semantic'][t].astype(np.int32)
        obs = semantic_sensor.predict(Obs(rgb=data['rgb'][t], depth=data['depth'][t]))

        pipeline.step(t, position, quat_wxyz, data['depth'][t].astype(np.float32), obs.semantic.astype(np.int32), data['rgb'][t].astype(np.uint8))
        
        mesh_vertices, mesh_colors, mesh_triangles = hydra_get_mesh(pipeline)
        rr_logger.log_mesh_data(mesh_vertices, mesh_colors, mesh_triangles)

        rr.log(f"{rr_logger.primary_camera_entity}/rgb", rr.Image(data['rgb'][t]).compress(jpeg_quality=95))
        rr.log(f"{rr_logger.primary_camera_entity}/semantic", rr.SegmentationImage(obs.semantic))
        rr.log(f"{rr_logger.primary_camera_entity}/instance", rr.SegmentationImage(obs.instance))
        rr.log(f"{rr_logger.primary_camera_entity}/depth", rr.DepthImage(data['depth'][t], meter=1.0))
        rr_logger.log_camera_tf(position, quat_wxyz)
        rr_logger.step()
    sg_sim.update()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / 'commands' / 'cfg' / f'{args.cfg_file}.yaml'
    cfg = OmegaConf.load(config_path)

    OmegaConf.resolve(cfg)
    main(cfg)