import csv
from tqdm import tqdm
import os
from omegaconf import OmegaConf
import pathlib
import click
import math

import numpy as np
import hydra_python as hydra
from hydra_python._plugins import habitat
from hydra_python import RRLogger

def load_eqa_data(cfg):
    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    
    # Filter to include only scenes with semantic annotations
    semantic_scenes = [f for f in os.listdir(cfg.semantic_annot_data_path) if os.path.isdir(os.path.join(cfg.semantic_annot_data_path, f))]

    filtered_question_data = []
    for data in questions_data:
        if data['scene'] in semantic_scenes:
            filtered_question_data.append(data)

    with open(cfg.init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }
    print(f"Loaded {len(filtered_question_data)} questions.")

    # init_pts = []
    # for data in filtered_question_data:
    #     scene_floor = data["scene"] + "_" + data["floor"]
    #     init_pts.append(init_pose_data[scene_floor]["init_pts"])

    # init_pts = np.array(init_pts)
    return filtered_question_data, init_pose_data

def initialize_hydra_pipeline(cfg, habitat_data, output_path):
    hydra.set_glog_level(cfg.glog_level, cfg.verbosity)
    configs = hydra.load_configs("habitat", labelspace_name=cfg.label_space)
    if not configs:
        click.secho(
            f"Invalid config: dataset 'habitat' and label space '{cfg.label_space}'",
            fg="red",
        )
        return
    pipeline_config = hydra.PipelineConfig(configs)
    pipeline_config.enable_reconstruction = True
    pipeline_config.label_names = {i: x for i, x in enumerate(habitat_data.colormap.names)}
    habitat_data.colormap.fill_label_space(pipeline_config.label_space) # TODO: check
    if output_path:
        pipeline_config.logs.log_dir = str(output_path)

    pipeline = hydra.HydraPipeline(
        pipeline_config, robot_id=0, config_verbosity=cfg.config_verbosity
    )
    pipeline.init(configs, hydra.create_camera(habitat_data.camera_info))

    if output_path:
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir()
        hydra.set_glog_dir(str(glog_dir))
    
    return pipeline

def log_traj_data_target_pose(rr_logger, poses, target_pose, poses_to_plot, orientations_to_plot, target_poses):
    poses_to_plot.extend([v[1] for v in poses])
    orientations_to_plot.extend([v[2] for v in poses])
    target_poses.append(target_pose)

    # for i in range(len(poses_to_plot)):
    #     rr_logger.log_agent_tf(poses_to_plot[i], orientations_to_plot[i])
    rr_logger.log_traj_data(poses_to_plot)
    rr_logger.log_target_poses(target_poses)

def main(cfg):
    questions_data, init_pose_data = load_eqa_data(cfg.data)

    output_path = hydra.resolve_output_path(cfg.output_path)

    for question_ind in tqdm(range(len(questions_data))):
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
            z_offset=cfg.habitat.z_offset)
        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
        
        rr_logger = RRLogger(question_path)

        # LOG NAVMESH
        # graph_nodes = [habitat_data.G.nodes[n]["pos"] for n in habitat_data.G]
        # positions_navmesh = np.array([hydra._plugins.habitat._habitat_to_world_eqa(p) for p in graph_nodes])
        # rr_logger.log_navmesh_data(positions_navmesh)

        # Extract initial pose
        scene_floor = question_data["scene"] + "_" + question_data["floor"]
        answer = question_data["answer"]
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]
        result = {"question_ind": question_ind}

        # Get max steps for current scene
        # scene_size = habitat_data.get_scene_size()
        # max_steps = int(math.sqrt(scene_size) * cfg.planner.max_step_room_size_ratio)
        max_steps=5
        # Get scene graph for current view
        poses = habitat_data.get_init_poses_eqa(init_pts, init_angle, cfg.habitat.camera_tilt_deg)
        hydra.run(
            pipeline,
            habitat_data,
            poses,
            show_progress=False,
            output_path=question_path,
            rr_logger=rr_logger,
        )
        vlm_planner = hydra.VLMPLannerEQA(questions_data[question_ind], question_path, pipeline, rr_logger)
        click.secho(f"Question:\n{vlm_planner._question}",fg="green",)
        
        poses_to_plot, orientations_to_plot, target_poses = [], [], []
        log_traj_data_target_pose(rr_logger, [poses[0]], poses[0][1], poses_to_plot, orientations_to_plot, target_poses)

        # poses_to_plot = [v[1] for v in poses]
        # orientations_to_plot = [v[2] for v in poses]
        # target_poses = [poses_to_plot[-1]]

        # rr_logger.log_agent_tf(poses_to_plot[0], orientations_to_plot[0])
        # rr_logger.log_traj_data([poses_to_plot[0]])
        # rr_logger.log_target_poses(target_poses)
        successes, successes_wo_done = 0, 0
        # Planner for current question
        for cnt_step in range(max_steps):
            # agent_state = habitat_data.get_state()
            click.secho(f"Planning at step: {vlm_planner.t}",fg="blue",)
            target_pose, done, confidence, answer_output = vlm_planner.get_next_action()
            import ipdb; ipdb.set_trace()
            if target_pose is not None:
                poses = habitat_data.get_trajectory_to_pose_world_eqa(target_pose, vlm_planner.sg_sim.navmesh_netx_graph)
                if poses is not None:
                    log_traj_data_target_pose(rr_logger, poses, target_pose, poses_to_plot, orientations_to_plot, target_poses)
                    rr_logger.log_text_data(vlm_planner.full_plan)

                    click.secho(f"Executing trajectory: {vlm_planner.t}",fg="yellow",)
                    hydra.run(
                        pipeline,
                        habitat_data,
                        poses,
                        show_progress=False,
                        output_path=question_path,
                        suffix=f't_{vlm_planner.t}',
                        rr_logger=rr_logger,
                    )
            if done and 'yes' in confidence:
                if answer == answer_output:
                    successes += 1
                    click.secho(f"Success at step {cnt_step} for {question_ind}:{scene_floor}",fg="blue",)
                    break
            if 'yes' in confidence:
                if answer == answer_output:
                    successes_wo_done += 1
                    click.secho(f"Success without done at step{cnt_step} for {question_ind}:{scene_floor}",fg="blue",)
                    break
        pipeline.save()

if __name__ == "__main__":
    cfg = OmegaConf.load('/home/saumyas/catkin_ws_semnav/src/hydra/python/src/hydra_python/commands/cfg/vlm_eqa.yaml')
    OmegaConf.resolve(cfg)
    main(cfg)