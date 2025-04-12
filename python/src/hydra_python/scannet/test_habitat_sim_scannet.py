from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os
from pathlib import Path

import numpy as np
import hydra_python as hydra
from hydra_python.run import run_eqa
from hydra_python._plugins import habitat
from hydra_python import RRLogger
from hydra_python.frontier_mapping_eqa.tsdf import TSDFPlanner
from hydra_python.frontier_mapping_eqa.utils import *
from hydra_python.frontier_mapping_eqa.geom import *

from hydra_python.utils import load_eqa_data, initialize_hydra_pipeline, get_instruction_from_eqa_data
from hydra_python.frontier_mapping_eqa.utils import pos_habitat_to_normal
import sys
import torch 

def main():

    habitat_cfg = {
        'scene_type' : 'scannet',
        'dataset_type': 'train',
        'sim_gpu': 0,
        'inflation_radius': 0.25,
        'img_width': 640,
        'img_height': 480,
        'camera_height': 1.5,
        'camera_tilt_deg': -30,
        'agent_z_offset': 0.,
        'hfov': 120,
        'z_offset': 0,
        'use_semantic_data': True,
    }
    habitat_cfg = OmegaConf.create(habitat_cfg)

    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    scene_name = '/mnt/hdd1/saumyas/data/semnav/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.glb'
    habitat_data = habitat.HabitatInterface(
        scene_name, 
        cfg=habitat_cfg,
        device=device,)
    # pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
    # pipeline.save()

if __name__ == "__main__":
    main()