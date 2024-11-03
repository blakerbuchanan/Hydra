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

import click
def main(stretch_parameter_file, hydra_cfg):

    # Need to define these arguments
    # Create robot
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
        hydra_cfg.output_path, 
        sensor_categories_mapping=sensor_categories_mapping
    )

    device = f"cuda:{hydra_cfg.gpu}" if torch.cuda.is_available() else "cpu"
    sg_sim = hydra.SceneGraphSim(
        hydra_cfg, 
        hydra_cfg.output_path, 
        hydra_pipeline, 
        rr_logger, 
        device=device)

    agent = RobotHydraAgent(
        robot, 
        parameters, 
        hydra_pipeline, 
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

    if 'gpt' in hydra_cfg.vlm.name.lower():
        vlm_planner = hydra.VLMPLannerEQAGPT(
            hydra_cfg.vlm,
            sg_sim,
            questions_data[question_ind], 
            question_path)
    elif 'gemini' in hydra_cfg.vlm.name.lower():
        vlm_planner = hydra.VLMPLannerEQAGemini(
            hydra_cfg.vlm,
            sg_sim,
            questions_data[question_ind], 
            question_path)
    else:
        raise NotImplementedError('VLM planner not implemented.')
    

    run_vlm_planner(
        manual_wait=False,
        max_planning_steps=hydra_cfg.planner.max_planning_steps,
        go_home_at_end=False
    )
    rotated = False
    succ = False
    for p_step in range(hydra_cfg.planner.max_planning_steps):
        click.secho(f"Planning step {p_step}",fg="green",)
        start = agent.robot.get_base_pose()

        start_is_valid = agent.space.is_valid(start, verbose=True)
        # if start is not valid move backwards a bit
        if not start_is_valid:
            click.secho(f"Start not valid. back up a bit.",fg="yellow",)
            ok = agent.recover_from_invalid_start()
            if ok:
                start = agent.robot.get_base_pose()
                start_is_valid = agent.space.is_valid(start, verbose=True)
            if not start_is_valid:
                click.secho(f"Failed to recover from invalid start state!",fg="red",)
                break
        
        (
            target_pose, 
            is_confident, 
            confidence_level, 
            answer_output
        ) = vlm_planner.get_next_action()
        agent.robot._rerun.log_text_data(vlm_planner.full_plan)
        if is_confident or confidence_level >= 0.9:
            succ = (answer == answer_output)
            if succ:
                successes += 1
                click.secho(f"Success at step{p_step}",fg="blue",)
                click.secho(f"VLM Planner answer: {answer_output}, Correct answer: {answer}",fg="blue",)
            else:
                click.secho(f"Failure at step {p_step}=",fg="red",)
                click.secho(f"VLM Planner answer: {answer_output}, Correct answer: {answer}",fg="red",)
            # break # TODO break at confidence or not in cfg
        else:
            if target_pose is not None:
    
    # succ = False

    #     target_pose, is_confident, confidence_level, answer_output = vlm_planner.get_next_action()
    #     if target_pose is not None:
    #         desired_path = [] # path_to_frontier will be voxel planner
            
    #         # target pose goes to path planner
    #         my_stretch.move_robot_and_start_hydra(x, y, theta)

    #     mesh_vertices, mesh_colors, mesh_triangles = hydra_get_mesh(self.pipeline)

    #     # Log to rerun
    #     rr_logger.log_mesh_data(mesh_vertices, mesh_colors, mesh_triangles)
    #     rr_logger.log_camera_tf(step.camera_data.pos, step.camera_data.rot)
    #     rr_logger.log_rosbag_img_data(step.camera_data)
    #     rr_logger.step()


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
