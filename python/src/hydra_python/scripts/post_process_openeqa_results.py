import pickle, json
import numpy as np
import csv, os

from omegaconf import OmegaConf

from hydra_python.utils import load_openeqa_data

if __name__ == "__main__":

    cfg_file = '/home/saumyas/catkin_ws_semnav/src/hydra/python/src/hydra_python/commands/cfg/vlm_openeqa_strange.yaml'
    cfg = OmegaConf.load(cfg_file)
    OmegaConf.resolve(cfg)

    metrics = {}

    # results_path = '/home/saumyas/catkin_ws_semnav/src/hydra/outputs/openEQA_gpt-4o-2024-08-06_grapheqa_single_floor_with_choices/'
    results_path = '/home/saumyas/catkin_ws_semnav/src/hydra/outputs/openEQA_gemini_grapheqa_single_floor_with_choices/'
    results_path = '/home/saumyas/catkin_ws_semnav/src/hydra/outputs/openEQA_Llama-4-Maverick-17B-128E-Instruct-FP8_grapheqa_single_floor_with_choices/'
    metrics_filename = results_path + 'metrics.json'
    questions_data, init_pose_data, choices_data = load_openeqa_data(cfg.data)

    # Save the data from the JSON results file
    # full_results_json_path = '/home/saumyas/catkin_ws_semnav/src/hydra/outputs/openEQA_gpt-4o-2024-08-06_grapheqa_single_floor_with_choices/gpt-4o-2024-08-06_images_True.json'
    # full_results_json_path = '/home/saumyas/catkin_ws_semnav/src/hydra/outputs/openEQA_gemini_grapheqa_single_floor_with_choices/gemini_images_True.json'
    full_results_json_path = '/home/saumyas/catkin_ws_semnav/src/hydra/outputs/openEQA_Llama-4-Maverick-17B-128E-Instruct-FP8_grapheqa_single_floor_with_choices/Llama-4-Maverick-17B-128E-Instruct-FP8_images_True.json'
    with open(full_results_json_path, 'r') as file:
        data = json.load(file)

    num_success, planning_steps_all_trajs, length_all_trajs = 0, 0, 0
    total_trajs = len(data.keys())

    for k, v in data.items():
        if v['Success']:
            num_success += 1
            length_all_trajs += v['metrics']['traj_length']
            planning_steps_all_trajs += v['metrics']['vlm_steps']

    metrics = {}
    metrics['length_all_trajs'] = length_all_trajs
    metrics['planning_steps_all_trajs'] = float(planning_steps_all_trajs)
    metrics['num_success'] = float(num_success)
    metrics['success_rate'] = float(num_success/total_trajs)
    metrics['total_trajs'] = total_trajs
    metrics['avg_traj_length'] = length_all_trajs / total_trajs
    # TODO(check this)
    metrics['avg_planning_steps'] = float(planning_steps_all_trajs/total_trajs)

    type_results = {}
    type_results['spatial understanding'] = 0
    type_results['object state recognition'] = 0
    type_results['functional reasoning'] = 0
    type_results['attribute recognition'] = 0
    type_results['world knowledge'] = 0
    type_results['object localization'] = 0
    type_results['object recognition'] = 0

    all_categories = ['spatial understanding', 'object state recognition', 'functional reasoning', 'attribute recognition', 'world knowledge', 'object localization', 'object recognition']
    for category in all_categories:
        type_results[f'{category}'] = 0
        type_results[f'{category} success'] = 0

    weighted_length_all_trajs = 0.
    max_length_all_trajs = 0.
    planning_steps_weighted_all_trajs = 0
    planning_steps_max_all_trajs = 0
    num_succ_weighted = 0
    num_succ_max = 0

    result_files = [f for f in os.listdir(results_path) if os.path.isfile(os.path.join(results_path, f)) and f.endswith('.pkl')]
    result_files = [os.path.join(results_path, f) for f in result_files]

    ques_ids_processed = []
        
    print('PROCESS QUESTION CATEGORIES')
    for question_ind, question_data in enumerate(questions_data):
        experiment_id = f'{question_ind}_{question_data["scene"]}'
        # print(experiment_id)

    metrics['type_results'] = type_results

    print(f"Saving file: {metrics_filename}")
    with open(metrics_filename, 'w') as file:
        json.dump(metrics, file, indent=4)
    print(f"Saved file: {metrics_filename}")