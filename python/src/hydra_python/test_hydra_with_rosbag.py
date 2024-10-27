from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os
from pathlib import Path

import numpy as np
import hydra_python as hydra
from hydra_python._plugins import habitat
from hydra_python import RRLogger
from hydra_python import TSDFPlanner
from hydra_python.frontier_mapping_eqa.utils import *
from hydra_python.frontier_mapping_eqa.geom import *

from hydra_python.utils import load_eqa_data, initialize_hydra_pipeline, get_instruction_from_eqa_data
from hydra_python.frontier_mapping_eqa.utils import pos_habitat_to_normal
import sys
import torch
import rosbag

class RobotData:
    def __init__(self, camera_info={}, translation=[], pos=[], rot=[]):
        self.pos = pos
        self.rot = rot
    def empty(self):
        if (len(self.pos) == 0 or len(self.rot) == 0):
            return True
        return False

class CameraData:
    def __init__(self, camera_info={}, rgb=[], depth=[], depth_encoding="", timestamp_s=0, timestamp_ns=0, pos=[], rot=[]):
        self.camera_info = camera_info
        self.rgb = rgb
        self.depth = depth
        self.depth_encoding = depth_encoding
        self.timestamp_s = timestamp_s
        self.timestamp_ns = timestamp_ns
        self.pos = pos
        self.rot = rot

    def empty(self):
        if (len(self.camera_info) == 0 or len(self.rgb) == 0 or len(self.depth) == 0 or
            len(depth_encoding) == 0 or len(self.pos) == 0 or len(self.rot) == 0):
            return True
   
        return False

class StepData:
    def __init__(self, camera_data=CameraData(), robot_data=RobotData()):
        self.camera_data = camera_data
        self.robot_data = robot_data

def main(cfg):

    bag = rosbag.Bag('/media/jmf1/73FF-DDD1/data/datasets/rll/semnav/stretch-virtual-folder/rosbags/data_vlnce/uninteresting.bag')
    steps = []
    
    for topic, msg, t in bag.read_messages(topics=['/camera/color/camera_info', '/camera/color/image_raw', '/tf']):
        step_data = StepData()

        try:
            # populate the camera info, but only once
            if topic == '/camera/color/camera_info':
                if len(camera_data.camera_info) == 0:
                    focal_length_x, width, focal_length_y, height = msg.K[0], msg.K[2], msg.K[4], msg.K[5]
                    camera_data.camera_info = {
                        "fx": float(focal_length_x),
                        "fy": float(focal_length_y),
                        "cx": float(width / 2.0),
                        "cy": float(height / 2.0),
                        "width": int(width),
                        "height": int(height),
                    }

            if topic == '/camera/color/image_raw':
                camera_data.timestamp_s = msg.header.stamp.secs
                camera_data.timestamp_s = msg.header.stamp.nsecs
                camera_data.rgb = msg.data

            if topic == '/camera/aligned_depth_to_color/image_raw':
                camera_data.depth = msg.data
                camera_data.depth_encoding = msg.encoding

            if topic == '/tf':
                for transform in msg.transforms:
                    if transform.header.frame_id == "base_link":
                        robot_data.pos = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                        robot_data.rot = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
                    if transform.header.frame_id == "base_link":
                        camera_data.pos = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                        camera_data.rot = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]

            
            if (robot_data.empty() == False and camera_data.empty() == False):
                print("Data full. Adding step.")
                step_data.robot_data = robot_data
                step_data.camera_data = camera_data
                steps.append(step_data)
                
                camera_data = CameraData()
                robot_data = RobotData()

        except Exception as e:
            # Code to handle the exception
            print("An error occurred:", e)

    import ipdb; ipdb.set_trace()
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / 'commands' / 'cfg' / f'{args.cfg_file}.yaml'
    cfg = OmegaConf.load(config_path)

    OmegaConf.resolve(cfg)
    main(cfg)
