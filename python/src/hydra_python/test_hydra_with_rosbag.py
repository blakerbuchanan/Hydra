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

from hydra_python.utils import load_eqa_data, initialize_hydra_pipeline_rosbag, get_instruction_from_eqa_data
from hydra_python.run import hydra_get_mesh
from hydra_python.frontier_mapping_eqa.utils import pos_habitat_to_normal
import sys
import torch
import rosbag
from collections import deque
from tqdm import tqdm

# Define a small threshold for synchronization tolerance (e.g., 0.05 seconds)
SYNC_THRESHOLD = 0.1

# Buffers for each topic
camera_info_buffer = deque()
camera_color_buffer = deque()
camera_depth_buffer = deque()
tf_buffer = deque()
camera_semantic_buffer = deque()

has_semantic_data = True

class RobotData:
    def __init__(self, camera_info={}, translation=[], pos=[], rot=[]):
        self.pos = pos
        self.rot = rot

    # Check if RobotData instance has all required data
    def empty(self):
        if (len(self.pos) == 0 or len(self.rot) == 0):
            return True
        return False

class CameraData:
    def __init__(self, camera_info={}, rgb=[], depth=[], depth_encoding="", semantic_image=[], timestamp_s=0, timestamp_ns=0, pos=[], rot=[]):
        self.camera_info = camera_info
        self.rgb = rgb
        self.depth = depth
        self.depth_encoding = depth_encoding
        self.semantic_image = semantic_image
        self.timestamp_s = timestamp_s
        self.timestamp_ns = timestamp_ns
        self.pos = pos
        self.rot = rot

    # Check if CameraData instance has all required data
    def empty(self):
        if (len(self.camera_info) == 0 or len(self.rgb) == 0 or len(self.depth) == 0 or
            len(self.depth_encoding) == 0 or len(self.pos) == 0 or len(self.rot) == 0):
            return True
   
        return False

class StepData:
    def __init__(self, camera_data=CameraData(), robot_data=RobotData()):
        self.camera_data = camera_data
        self.robot_data = robot_data

def process_if_synchronized():

    steps_out = []
    while camera_info_buffer and camera_color_buffer and camera_depth_buffer and tf_buffer and camera_semantic_buffer:
        # Get the earliest message in each buffer
        camera_info_msg, camera_info_time = camera_info_buffer[0]
        camera_color_msg, camera_color_time = camera_color_buffer[0]
        camera_depth_msg, camera_depth_time = camera_depth_buffer[0]
        camera_semantic_msg, camera_semantic_time = camera_semantic_buffer[0]
        tf_msg, tf_time = tf_buffer[0]

        # Check if they are within the sync threshold
        # NOTE this does not check all combinations of time...and that's probably fine, but idk
        if (abs(camera_info_time - camera_color_time) < SYNC_THRESHOLD and 
           abs(camera_color_time - tf_time) < SYNC_THRESHOLD and
           abs(camera_color_time - camera_depth_time) < SYNC_THRESHOLD):
        # if True:
            camera_data = CameraData()
            robot_data = RobotData()
            step_data = StepData()

            # populate the camera info, but only once
            if len(camera_data.camera_info) == 0:
                focal_length_x, width, focal_length_y, height = camera_info_msg.K[0], camera_info_msg.K[2], camera_info_msg.K[4], camera_info_msg.K[5]
                camera_data.camera_info = {
                    "fx": float(focal_length_x),
                    "fy": float(focal_length_y),
                    "cx": float(width / 2.0),
                    "cy": float(height / 2.0),
                    "width": int(width),
                    "height": int(height),
                }
                
            camera_data.timestamp_s = camera_color_msg.header.stamp.secs
            camera_data.timestamp_ns = camera_color_msg.header.stamp.nsecs
            camera_data.rgb = convertRgbToNumpy(camera_color_msg)

            camera_data.depth = convertDepthToNumpy(camera_depth_msg)
            camera_data.depth_encoding = camera_depth_msg.encoding

                
            for transform in tf_msg.transforms:
                if transform.header.frame_id == "base_link":
                    robot_data.pos = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                    robot_data.rot = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
                if transform.header.frame_id == "link_mast":
                    camera_data.pos = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                    camera_data.rot = [transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]
            
            if (has_semantic_data == True):
                camera_data.semantic_image = convertRgbToNumpy(camera_semantic_msg.image)

            step_data.robot_data = robot_data
            step_data.camera_data = camera_data

            # we need to get these images outside of this function before we pop them...

            steps_out.append(step_data)

            # Remove the processed messages
            camera_info_buffer.popleft()
            camera_color_buffer.popleft()
            camera_depth_buffer.popleft()
            camera_semantic_buffer.popleft()
            tf_buffer.popleft()

        else:
            # Remove the earliest message that is out of sync
            earliest_time = min(camera_info_time, camera_color_time, camera_depth_time, camera_semantic_time, tf_time)
            if camera_info_time == earliest_time:
                camera_info_buffer.popleft()
            elif camera_color_time == earliest_time:
                camera_color_buffer.popleft()
            elif camera_depth_time == earliest_time:
                camera_depth_buffer.popleft()
            elif has_semantic_data == True and camera_semantic_time == earliest_time:
                camera_semantic_buffer.popleft()
            else:
                tf_buffer.popleft()

    return steps_out

# Helper functions for image conversion
def convertRgbToNumpy(msg):
    image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    return image_np

def convertDepthToNumpy(msg):
    if msg.encoding == "16UC1":
        depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        depth_array = depth_array.astype(np.float32) / 1000.0  # Convert mm to meters if necessary
    elif msg.encoding == "32FC1":
        depth_array = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

    return depth_array

def main(cfg):

    # bag = rosbag.Bag('/media/jmf1/73FF-DDD1/data/datasets/rll/semnav/stretch-virtual-folder/rosbags/data_vlnce/202410291217-explore_small_big_lab.bag.bag')
    bag = rosbag.Bag('/home/saumyas/semnav_workspace/semantic_data_small_big_lab.bag')
    
    steps = []
    
    for topic, msg, t in bag.read_messages(topics=['/camera/color/camera_info', '/camera/color/image_raw', 'camera/semantic/image_raw_test', '/camera/aligned_depth_to_color/image_raw', '/tf']):
        if topic == '/camera/color/camera_info':
            camera_info_buffer.append((msg, t.to_sec()))
        elif topic == '/camera/color/image_raw':
            camera_color_buffer.append((msg, t.to_sec()))
        elif topic == '/camera/aligned_depth_to_color/image_raw':
            camera_depth_buffer.append((msg, t.to_sec()))
        elif topic == 'camera/semantic/image_raw_test':
            camera_semantic_buffer.append((msg, t.to_sec()))
        elif topic == '/tf':
            tf_buffer.append((msg, t.to_sec()))
        
        # Check if any data is synchronized
        steps_out = process_if_synchronized()

        # Check for synchronization success and that both robot and camera data are populated
        if len(steps_out) != 0:
            for step in steps_out:
                if (step.robot_data.empty() == False and step.camera_data.empty() == False):
                    steps.append(step)

    output_path = cfg.output_path
    os.makedirs(cfg.output_path, exist_ok=True)
    output_path = Path(cfg.output_path)

    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    pipeline = initialize_hydra_pipeline_rosbag(cfg.hydra, steps[0].camera_data.camera_info, output_path)
    rr_logger = RRLogger(output_path)

    click.secho(f'Location: Bosch Pittsburgh lab',fg="green",)

    for step in tqdm(steps):
        labels = np.ones(step.camera_data.depth.shape, dtype=np.uint8)
        pipeline.step(step.camera_data.timestamp_s, step.camera_data.pos, step.camera_data.rot, step.camera_data.depth, labels, step.camera_data.rgb)
        mesh_vertices, mesh_colors, mesh_triangles = hydra_get_mesh(pipeline)
        rr_logger.log_mesh_data(mesh_vertices, mesh_colors, mesh_triangles)
        rr_logger.log_camera_tf(step.camera_data.pos, step.camera_data.rot)
        rr_logger.log_rosbag_img_data(step.camera_data)

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
