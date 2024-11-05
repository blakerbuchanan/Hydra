from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os
from pathlib import Path

import numpy as np
import hydra_python as hydra

from hydra_python.utils import hydra_get_mesh, initialize_hydra_pipeline_stretch
import sys
import torch
from collections import deque
from tqdm import tqdm

from hydra_python.stretch_ai_utils.robot_hydra_agent import RobotHydraAgent
from hydra_python.stretch_ai_utils.utils import write_config_yaml, write_to_pickle


from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.perception import create_semantic_sensor


def main(stretch_parameter_file, hydra_cfg):

    # Need to define these arguments
    # Create robot

    os.makedirs(hydra_cfg.output_path, exist_ok=True)
    output_path = Path(hydra_cfg.output_path)


    parameters = get_parameters(stretch_parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=parameters.data['robot_ip'],
        use_remote_computer=True,
        parameters=parameters,
        enable_rerun_server=parameters.data['enable_rerun_server'],
        publish_observations=parameters.data['enable_realtime_updates'],
    )

    # TODO get semantic data
    print("- Create semantic sensor based on detic")
    device_id = parameters.data['device_id_sem_sensor']
    if parameters.data['use_semantic_sensor']:
        semantic_sensor = create_semantic_sensor(
            parameters=parameters,
            device_id=device_id,
            verbose=True,
        )
        sensor_categories_mapping = semantic_sensor.seg_id_to_name
        # write_config_yaml(sensor_categories_mapping)
        # import ipdb; ipdb.set_trace()
    else:
        semantic_sensor = None
        sensor_categories_mapping = None
        
    obs = robot.get_observation()
    # output_path = parameters.data['output_path']
    hydra_pipeline = initialize_hydra_pipeline_stretch(
        hydra_cfg.hydra, 
        obs, 
        output_path,
        sensor_categories_mapping=sensor_categories_mapping
    )

    device = f"cuda:{hydra_cfg.gpu}" if torch.cuda.is_available() else "cpu"
    sg_sim = hydra.SceneGraphSim(
        hydra_cfg, 
        output_path, 
        hydra_pipeline, 
        rr_logger=None, 
        device=device)

    agent = RobotHydraAgent(
        robot, 
        parameters, 
        hydra_pipeline, 
        sg_sim,
        semantic_sensor, 
        enable_realtime_updates=parameters.data['enable_realtime_updates']
    )
    agent.start()
    agent.update()

    if parameters["agent"]["in_place_rotation_steps"] > 0:
        agent.rotate_in_place(
            steps=parameters["agent"]["in_place_rotation_steps"],
            visualize=False,
        )

    # print("============writing pickle file")
    # write_to_pickle(agent.obs_history, 'data_with_semantics')
    # click.secho(f'Location: Bosch Pittsburgh lab',fg="green",)
    
    manual_wait = False
    agent.run_exploration(
        manual_wait,
        explore_iter=parameters["exploration_steps"],
        task_goal=None,
        random_goals=False,
        go_home_at_end=False,
        visualize=False,
    )
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    stretch_config_path = str(Path(__file__).resolve().parent) + f'/stretch_ai_utils/cfg/{args.cfg_file}.yaml'

    hydra_cfg_path = Path('/home/saumyas/semnav_workspace/src/hydra/python/src/hydra_python/commands/cfg/vlm_eqa_stretch.yaml')
    hydra_cfg = OmegaConf.load(hydra_cfg_path)
    OmegaConf.resolve(hydra_cfg)

    main(stretch_config_path, hydra_cfg)
