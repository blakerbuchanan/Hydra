from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import hydra_python as hydra
from hydra_python._plugins import habitat
from hydra_python import RRLogger
from hydra_python import TSDFPlanner
from hydra_python.frontier_mapping_eqa.utils import *
from hydra_python.frontier_mapping_eqa.geom import *

from hydra_python.utils import load_eqa_data, initialize_hydra_pipeline
from hydra_python.frontier_mapping_eqa.utils import pos_habitat_to_normal

def main(cfg):
    questions_data, init_pose_data = load_eqa_data(cfg.data)

    output_path = cfg.output_path
    os.makedirs(cfg.output_path, exist_ok=True)
    output_path = Path(cfg.output_path)
    # output_path = hydra.resolve_output_path(cfg.output_path)

    successes, successes_wo_done = 0, 0
    for question_ind in tqdm(range(len(questions_data))):
        if question_ind in [0,1]:
            continue
        question_data = questions_data[question_ind]
        print(f'\n========\nIndex: {question_ind} Scene: {question_data["scene"]} Floor: {question_data["floor"]}')

        # Planner reset with the new quesion
        question_path = hydra.resolve_output_path(output_path / f'{question_ind}_{question_data["scene"]}')
        scene_name = f'{cfg.data.scene_data_path}/{question_data["scene"]}/{question_data["scene"][6:]}.basis.glb'
        habitat_data = habitat.HabitatInterface(
            scene_name, 
            scene_type=cfg.habitat.scene_type, 
            camera_height=cfg.habitat.camera_height,
            width=cfg.habitat.img_width, 
            height=cfg.habitat.img_height,
            agent_z_offset=cfg.habitat.agent_z_offset,
            hfov=cfg.habitat.hfov,
            z_offset=cfg.habitat.z_offset,
            camera_tilt=cfg.habitat.camera_tilt_deg*np.pi/180)
        cfg_hydra_copy = OmegaConf.create(OmegaConf.to_container(cfg.hydra, resolve=True))
        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
        
        rr_logger = RRLogger(question_path)

        # Extract initial pose
        scene_floor = question_data["scene"] + "_" + question_data["floor"]
        answer = question_data["answer"]
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]

        # Setup TSDF planner
        pts_normal = pos_habitat_to_normal(init_pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(habitat_data.pathfinder, floor_height)
        cam_intr = get_cam_intr(cfg.habitat.hfov, cfg.habitat.img_height, cfg.habitat.img_width)
        
        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            cfg=cfg.frontier_mapping,
            vol_bnds=tsdf_bnds,
            cam_intr=cam_intr,
            floor_height_offset=0,
            pts_init=pts_normal,
            rr_logger=rr_logger,
        )

        # Get poses for hydra at init view
        poses = habitat_data.get_init_poses_eqa(init_pts, init_angle, cfg.habitat.camera_tilt_deg)
        # Get scene graph for init view
        hydra.run_eqa(
            pipeline,
            habitat_data,
            poses,
            output_path=question_path,
            rr_logger=rr_logger,
            tsdf_planner=tsdf_planner,
            save_image=cfg.vlm.use_image,
        )

        vlm_planner = hydra.VLMPLannerEQA(
            cfg.vlm,
            questions_data[question_ind], 
            question_path, 
            pipeline, 
            rr_logger, 
            tsdf_planner.frontier_to_sample_normal,)
        click.secho(f"Question:\n{vlm_planner._question} \n Answer: {answer}",fg="green",)

        num_steps = 100
        for cnt_step in range(num_steps):
            target_pose, done, confidence, answer_output = vlm_planner.get_next_action()
            rr_logger.log_text_data(vlm_planner.full_plan)
            if done and 'yes' in confidence:
                if answer == answer_output:
                    successes += 1
                    click.secho(f"Success at step {cnt_step} for {question_ind}:{scene_floor}",fg="blue",)
                else:
                    click.secho(f"Failure at step {cnt_step} for {question_ind}:{scene_floor}",fg="blue",)
                break
            elif 'yes' in confidence:
                if answer == answer_output:
                    successes_wo_done += 1
                    click.secho(f"Success without done at step{cnt_step} for {question_ind}:{scene_floor}",fg="blue",)
                else:
                    click.secho(f"Failure without done at step {cnt_step} for {question_ind}:{scene_floor}",fg="blue",)
                break
            else:
                if target_pose is not None:
                    # desired_path = tsdf_planner.sample_frontier()
                    current_heading = habitat_data.get_heading_angle()
                    desired_path = tsdf_planner.path_to_frontier(target_pose)

                    agent = habitat_data._sim.get_agent(0)  # Assuming agent ID 0
                    current_pos = agent.get_state().position
                    frontier_habitat = pos_normal_to_habitat(target_pose)
                    frontier_habitat[1] = current_pos[1]
                    path = habitat_sim.nav.ShortestPath()
                    path.requested_start = current_pos
                    path.requested_end = frontier_habitat
                    # Compute the shortest path
                    found_path = habitat_data.pathfinder.find_path(path)
                    if found_path:
                        desired_path = pos_habitat_to_normal(np.array(path.points))
                        rr_logger.log_traj_data(desired_path)
                        rr_logger.log_target_poses(target_pose)
                    else:
                        click.secho(f"Cannot find navigable path: {cnt_step}",fg="red",)
                        continue

                    poses = habitat_data.get_trajectory_from_path_habitat_frame2(desired_path, current_heading, cfg.habitat.camera_tilt_deg)
                    if poses is not None:
                        click.secho(f"Executing trajectory: {vlm_planner.t}",fg="yellow",)
                        hydra.run_eqa(
                            pipeline,
                            habitat_data,
                            poses,
                            output_path=question_path,
                            rr_logger=rr_logger,
                            tsdf_planner=tsdf_planner,
                            vlm_planner=vlm_planner,
                            save_image=cfg.vlm.use_image,
                        )

        pipeline.save()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR]: Configuration path not provided. Please provide a path to configuration file for the VLM planner.")
        sys.exit(1)  # Exit the script with an error code

    config_path = sys.argv[1]
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

    OmegaConf.resolve(cfg)
    main(cfg)