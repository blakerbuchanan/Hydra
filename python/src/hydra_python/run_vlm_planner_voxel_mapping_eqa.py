from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import click

from voxel_mapping.voxel import SparseVoxelMap, SparseVoxelMapNavigationSpace, plan_to_goal
from voxel_mapping import get_parameters, Observations
from voxel_mapping.motion.algo import RRTConnect, Shortcut, SimplifyXYT

import hydra_python as hydra
from hydra_python._plugins import habitat
from hydra_python import RRLogger

from hydra_python.frontier_mapping_eqa.geom import get_cam_intr
from hydra_python.utils import load_eqa_data, initialize_hydra_pipeline
from hydra_python.frontier_mapping_eqa.utils import get_cam_pose_tsdf, pos_habitat_to_normal


def main(habitat_cfg, mapping_cfg_path):

    questions_data, init_pose_data = load_eqa_data(habitat_cfg.data)
    output_path = hydra.resolve_output_path(habitat_cfg.output_path)

    successes, successes_wo_done = 0, 0
    for question_ind in tqdm(range(len(questions_data))):
        # if question_ind==0:
        #     continue
        question_data = questions_data[question_ind]
        print(f'\n========\nIndex: {question_ind} Scene: {question_data["scene"]} Floor: {question_data["floor"]}')

        # Planner reset with the new quesion
        question_path = hydra.resolve_output_path(output_path / f'{question_ind}_{question_data["scene"]}')
        scene_name = f'{habitat_cfg.data.scene_data_path}/{question_data["scene"]}/{question_data["scene"][6:]}.basis.glb'
        habitat_data = habitat.HabitatInterface(
            scene_name, 
            scene_type=habitat_cfg.habitat.scene_type, 
            camera_height=habitat_cfg.habitat.camera_height,
            width=habitat_cfg.habitat.img_width, 
            height=habitat_cfg.habitat.img_height,
            agent_z_offset=habitat_cfg.habitat.agent_z_offset,
            hfov=habitat_cfg.habitat.hfov,
            z_offset=habitat_cfg.habitat.z_offset,
            camera_tilt=habitat_cfg.habitat.camera_tilt_deg*np.pi/180)
        pipeline = initialize_hydra_pipeline(habitat_cfg.hydra, habitat_data, question_path)
        
        rr_logger = RRLogger(question_path)

        # Extract initial pose
        scene_floor = question_data["scene"] + "_" + question_data["floor"]
        answer = question_data["answer"]
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]

        cam_intr = get_cam_intr(habitat_cfg.habitat.hfov, habitat_cfg.habitat.img_height, habitat_cfg.habitat.img_width)

        parameters = get_parameters(mapping_cfg_path)

        voxel_map = SparseVoxelMap.from_parameters(
            parameters,
            voxel_size=parameters['voxel_size'],
            use_instance_memory=False,
            rr_logger=rr_logger,
        )

        # Create planning space
        space = SparseVoxelMapNavigationSpace(
            voxel_map,
            step_size=parameters["motion_planner"]["step_size"],
            rotation_step_size=parameters["motion_planner"]["rotation_step_size"],
            dilate_frontier_size=parameters["motion_planner"]["frontier"]["dilate_frontier_size"],
            dilate_obstacle_size=parameters["motion_planner"]["frontier"]["dilate_obstacle_size"],
            grid=voxel_map.grid,
            cam_intr=cam_intr
        )

        planner = RRTConnect(space, space.is_valid)
        if parameters["motion_planner"]["shortcut_plans"]:
            planner = Shortcut(planner, parameters["motion_planner"]["shortcut_iter"])
        if parameters["motion_planner"]["simplify_plans"]:
            planner = SimplifyXYT(planner, min_step=0.05, max_step=1.0, num_steps=8)

        # final_pts = np.array([7, init_pts[1], 8.0])
        # traj = np.linspace(np.array(init_pts), final_pts, num=10)
        # Get poses for hydra at init view
        poses = habitat_data.get_init_poses_eqa(init_pts, init_angle, habitat_cfg.habitat.camera_tilt_deg)
        # Get scene graph for init view
        hydra.run_eqa(
            pipeline,
            habitat_data,
            poses,
            output_path=question_path,
            rr_logger=rr_logger,
            voxel_space=space,
        )

        vlm_planner = hydra.VLMPLannerEQA(
            questions_data[question_ind], 
            question_path, pipeline, 
            rr_logger, 
            space.outside_frontier_points)
        
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
                    click.secho(f"Failire at step {cnt_step} for {question_ind}:{scene_floor}",fg="blue",)
                break
            elif 'yes' in confidence:
                if answer == answer_output:
                    successes_wo_done += 1
                    click.secho(f"Success without done at step{cnt_step} for {question_ind}:{scene_floor}",fg="blue",)
                else:
                    click.secho(f"Failire without done at step {cnt_step} for {question_ind}:{scene_floor}",fg="blue",)
                break
            else:
                if target_pose is not None:
                    pts_normal = habitat_data.get_state(is_eqa=True)[0]
                    current_heading = habitat_data.get_heading_angle()
                    base_pose = np.append(pts_normal[:2], current_heading)

                    res = plan_to_goal(base_pose, target_pose, planner, space, verbose=True)
                    if not res.success:
                        continue
                    path = np.array([pt.state for pt in res.trajectory])
                    path_xyz = np.concatenate([path[:,:2], np.full((path.shape[0],1), pts_normal[2])],1)
                    
                    rr_logger.log_traj_data(path_xyz)
                    rr_logger.log_target_poses(np.append(target_pose[:2], pts_normal[2]))
                    poses = habitat_data.get_trajectory_from_path_angles_habitat_frame(path_xyz, path[:,2], current_heading, habitat_cfg.habitat.camera_tilt_deg)
                    current_heading = path[-1,2]
                    click.secho(f"Executing plan of length {len(poses)}.",fg="green",)
                    hydra.run_eqa(
                        pipeline,
                        habitat_data,
                        poses,
                        output_path=question_path,
                        rr_logger=rr_logger,
                        voxel_space=space,
                        vlm_planner=vlm_planner,
                    )
        pipeline.save()

if __name__ == "__main__":
    habitat_cfg = OmegaConf.load('/home/saumyas/catkin_ws_semnav/src/hydra/python/src/hydra_python/commands/cfg/vlm_eqa.yaml')
    OmegaConf.resolve(habitat_cfg)

    mapping_cfg_path = '/home/saumyas/catkin_ws_semnav/src/hydra/python/src/hydra_python/commands/cfg/voxel_mapping.yaml'
    mapping_cfg = OmegaConf.load(mapping_cfg_path)
    OmegaConf.resolve(mapping_cfg)

    main(habitat_cfg, mapping_cfg_path)