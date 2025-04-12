from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os, time
import sys, json
from pathlib import Path
from copy import deepcopy

import numpy as np
import hydra_python as hydra
from hydra_python.run import run_eqa
from hydra_python._plugins import habitat
from hydra_python import RRLogger
from hydra_python.frontier_mapping_eqa.tsdf import TSDFPlanner
from hydra_python.frontier_mapping_eqa.utils import *
from hydra_python.frontier_mapping_eqa.geom import *

from hydra_python.utils import load_openeqa_data, initialize_hydra_pipeline, get_instruction_from_eqa_data, get_traj_len_from_poses, check_if_multifloor
from hydra_python.frontier_mapping_eqa.utils import pos_habitat_to_normal
import torch

def load_experiment_data(filename='experiment_results.json'):
    if not os.path.exists(filename):
        data={}
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
    
    with open(filename, 'r') as file:
        return json.load(file)

def save_experiment_data(data=None, filename='experiment_status.json'):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def log_experiment_status(experiment_id, success, metrics=None, filename='experiment_status.json'):
    data = load_experiment_data(filename)
    data[experiment_id] = {"Success": success}
    if metrics:
        data[experiment_id]["metrics"] = metrics
    save_experiment_data(data, filename)

def should_skip_experiment(experiment_id, filename='experiment_status.json'):
    data = load_experiment_data(filename)
    return experiment_id in [k.split('_')[1] for k in data.keys()]

def main(cfg):
    questions_data, init_pose_data, choices_data = load_openeqa_data(cfg.data)

    output_path = cfg.output_path
    os.makedirs(cfg.output_path, exist_ok=True)
    output_path = Path(cfg.output_path)
    results_filename = output_path / f'{cfg.results_filename}.json'
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    if not cfg.data.use_semantic_data:
        from hydra_python.detection.detic_segmenter import DeticSegmenter
        segmenter = DeticSegmenter(cfg)
    else:
        segmenter = None
    
    # evaluation_prompt = load_evaluation_prompt(cfg.data.evaluation_prompt_file)

    successes = 0
    for question_ind in tqdm(range(len(questions_data))):
        # if question_ind not in [25, 29, 33, 34, 38, 45, 49, 52, 57, 59, 63, 76, 81, 82, 88, 100, 104, 106, 107]:
        if question_ind not in [52, 63, 76, 79, 100]:
            continue
        question_data = questions_data[question_ind]
        question_id = question_data["question_id"]
        scene = init_pose_data[question_data['episode_history']]["scene_id"]
        choices = choices_data[question_id]['choices']

        answer = choices_data[question_id]["answer_id"]
        experiment_id = f'{question_ind}_{question_id}'

        if should_skip_experiment(question_data["question_id"], filename=results_filename):
            click.secho(f'Skipping==Index: {question_ind} Scene: {question_data["question_id"]}=======',fg="yellow",)
            continue
        else:
            click.secho(f'Executing=========Index: {question_ind} Scene: {question_data["question_id"]}=======',fg="green",)

        # Planner reset with the new quesion
        question_path = hydra.resolve_output_path(output_path / experiment_id)
        scene_name = f'{cfg.data.scene_data_path}/{scene}'
                
        habitat_data = habitat.HabitatInterface(
            scene_name, 
            cfg=cfg.habitat,
            device=device,)
        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
        
        rr_logger = RRLogger(question_path)

        # Extract initial pose
        init_pts = init_pose_data[question_data['episode_history']]["init_pos"]
        init_angle = init_pose_data[question_data['episode_history']]["init_angle"]

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

        sg_sim = hydra.SceneGraphSim(
            cfg, 
            question_path, 
            pipeline, 
            rr_logger, 
            device=device, 
            clean_ques_ans=question_data["question"],
            enrich_object_labels=choices_data[question_id]['enrich_labels'])

        # Get poses for hydra at init view
        poses = habitat_data.get_init_poses_eqa(init_pts, init_angle, cfg.habitat.camera_tilt_deg)
        # Get scene graph for init view
        run_eqa(
            pipeline,
            habitat_data,
            poses,
            output_path=question_path,
            rr_logger=rr_logger,
            tsdf_planner=tsdf_planner,
            sg_sim=sg_sim,
            save_image=cfg.vlm.use_image,
            segmenter=segmenter,
        )

        if 'gpt' in cfg.vlm.name.lower():
            vlm_planner = hydra.VLMPLannerEQAGPT(
                cfg.vlm,
                sg_sim,
                question_data["question"], 
                ["A", "B", "C", "D"], 
                choices, 
                answer, 
                question_path)
        elif 'gemini' in cfg.vlm.name.lower():
            vlm_planner = hydra.VLMPLannerEQAGemini(
                cfg.vlm,
                sg_sim,
                question_data["question"], 
                ["A", "B", "C", "D"], 
                choices, 
                answer, 
                question_path)
        else:
            raise NotImplementedError('VLM planner not implemented.')

        # LOG NAVMESH
        graph_nodes = np.array([habitat_data.G.nodes[n]["pos"].flatten() for n in habitat_data.G])
        rr_logger.log_navmesh_data(pos_habitat_to_normal(graph_nodes))
        # LOG data traj
        data_traj = pos_habitat_to_normal(np.array(init_pose_data[question_data['episode_history']]["full_traj_pos"]))
        rr_logger.log_traj_data(data_traj, name='data_traj')
        is_multifloor = False if not cfg.data.use_multifloor_questions else check_if_multifloor(data_traj[:,2])

        click.secho(f'Index:{question_ind} Scene: {question_data["question_id"]}',fg="green",)
        click.secho(f"Question:\n{vlm_planner._question} \n Answer: {answer}",fg="green",)

        num_steps = 30
        succ = False
        planning_steps = 0
        traj_length = 0.
        for cnt_step in range(num_steps):
            start = time.time()
            target_pose, target_id, is_confident, confidence_level, answer_output = vlm_planner.get_next_action()
            click.secho(f"VLM planning time for overall step {cnt_step} and vlm step {planning_steps} is {time.time()-start}",fg="green",)
            
            if is_confident or (confidence_level>0.85):
                succ = (answer == answer_output)
                if succ:
                    successes += 1
                    result = f"Success at vlm step{planning_steps} for {experiment_id}"
                    click.secho(result,fg="blue",)
                    click.secho(f"VLM Planner answer: {answer_output}, Correct answer: {answer}",fg="blue",)
                else:
                    result = f"Failure at vlm step {planning_steps} for {experiment_id}"
                    click.secho(result,fg="red",)
                    click.secho(f"VLM Planner answer: {answer_output}, Correct answer: {answer}",fg="red",)
                rr_logger.log_text_data(vlm_planner.full_plan + "\n" + result)
                break

            if target_pose is not None:

                # desired_path = tsdf_planner.sample_frontier()
                current_heading = habitat_data.get_heading_angle()
                # desired_path = tsdf_planner.path_to_frontier(target_pose) # not being used anymore

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
                    desired_path = pos_habitat_to_normal(np.array(path.points)[:-1])
                    rr_logger.log_traj_data(desired_path)
                    rr_logger.log_target_poses(target_pose)
                else:
                    click.secho(f"Cannot find navigable path at {cnt_step}. Continuing..",fg="red",)
                    continue

                poses = habitat_data.get_trajectory_from_path_habitat_frame2(target_pose, desired_path, current_heading, cfg.habitat.camera_tilt_deg)
                if poses is not None:
                    click.secho(f"Executing trajectory at overall step {cnt_step} and vlm step {planning_steps}",fg="yellow",)
                    run_eqa(
                        pipeline,
                        habitat_data,
                        poses,
                        output_path=question_path,
                        rr_logger=rr_logger,
                        tsdf_planner=tsdf_planner,
                        sg_sim=sg_sim,
                        save_image=cfg.vlm.use_image,
                        segmenter=segmenter,
                    )
                    traj_length += get_traj_len_from_poses(poses)

                    ## If trajectory successfully executed
                    rr_logger.log_text_data(vlm_planner.full_plan)
                    planning_steps+=1
                else:
                    click.secho(f"Cannot find trajectory from navigable path at {cnt_step}. Continuing..",fg="red",)
                    continue
            else:
                click.secho(f"VLM planner failed at overall step {cnt_step}. Continuing...",fg="red",)
        
        metrics = {
            'vlm_steps': planning_steps,
            'overall_steps': cnt_step,
            'is_confident': is_confident,
            'confidence_level': confidence_level,
            'traj_length': traj_length,
            'category': question_data["category"],
            'question': question_data["question"],
            'answer': question_data["answer"],
            'answer_id': answer,
            'choices': choices,
            'answer_output': answer_output,
            'is_multifloor': is_multifloor,
        }
        log_experiment_status(experiment_id, succ, metrics=metrics, filename=results_filename)
        habitat_data._sim.close(destroy=True)
        pipeline.save()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent.parent / 'commands' / 'cfg' / f'{args.cfg_file}.yaml'
    cfg = OmegaConf.load(config_path)

    OmegaConf.resolve(cfg)
    main(cfg)