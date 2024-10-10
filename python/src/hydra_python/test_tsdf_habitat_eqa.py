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

from hydra_python.utils import load_eqa_data, initialize_hydra_pipeline
from hydra_python.frontier_mapping_eqa.utils import pos_habitat_to_normal

def main(cfg):
    questions_data, init_pose_data = load_eqa_data(cfg.data)

    output_path = cfg.output_path
    os.makedirs(cfg.output_path, exist_ok=True)
    output_path = Path(cfg.output_path)
    # output_path = hydra.resolve_output_path(cfg.output_path)

    for question_ind in tqdm(range(len(questions_data))):
        if question_ind==0:
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
        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
        
        rr_logger = RRLogger(question_path)

        # Extract initial pose
        scene_floor = question_data["scene"] + "_" + question_data["floor"]
        answer = question_data["answer"]
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]
        # init_pts[0] = -0.154
        # init_pts[2] = 0.69

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
        )

        # LOG NAVMESH
        graph_nodes = np.array([habitat_data.G.nodes[n]["pos"] for n in habitat_data.G]).squeeze()
        positions_navmesh = np.array([pos_habitat_to_normal(p) for p in graph_nodes])
        rr_logger.log_navmesh_data(positions_navmesh)

        vlm_planner = hydra.VLMPLannerEQA(
            questions_data[question_ind], 
            question_path, 
            pipeline, 
            rr_logger, 
            tsdf_planner.frontier_to_sample_normal)
        click.secho(f"Question:\n{vlm_planner._question} \n Answer: {answer}",fg="green",)

        num_steps = 200
        for i in range(num_steps):
            current_heading = habitat_data.get_heading_angle()
            desired_path, frontier_normal = tsdf_planner.sample_frontier()
            
            # # Create a path object
            agent = habitat_data._sim.get_agent(0)  # Assuming agent ID 0
            current_pos = agent.get_state().position
            frontier_habitat = pos_normal_to_habitat(frontier_normal)
            frontier_habitat[1] = current_pos[1]
            path = habitat_sim.nav.ShortestPath()
            path.requested_start = current_pos
            path.requested_end = frontier_habitat
            # Compute the shortest path
            found_path = habitat_data.pathfinder.find_path(path)
            if found_path:
                desired_path = pos_habitat_to_normal(np.array(path.points))
                rr_logger.log_traj_data(desired_path)
                rr_logger.log_target_poses(frontier_normal)
            else:
                click.secho(f"Cannot find navigable path: {i}",fg="red",)
                continue

            poses = habitat_data.get_trajectory_from_path_habitat_frame2(desired_path, current_heading, cfg.habitat.camera_tilt_deg)
            click.secho(f"Executing trajectory: {i}",fg="yellow",)
            hydra.run_eqa(
                pipeline,
                habitat_data,
                poses,
                output_path=question_path,
                rr_logger=rr_logger,
                tsdf_planner=tsdf_planner,
                vlm_planner=vlm_planner,
            )
        pipeline.save()

if __name__ == "__main__":
    cfg = OmegaConf.load('/home/saumyas/catkin_ws_semnav/src/hydra/python/src/hydra_python/commands/cfg/vlm_eqa.yaml')
    OmegaConf.resolve(cfg)
    main(cfg)