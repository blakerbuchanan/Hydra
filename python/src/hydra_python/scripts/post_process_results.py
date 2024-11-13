import pickle, json
import numpy as np

if __name__ == "__main__":
    filepath2 = '/home/saumyas/catkin_ws_semnav/src/hydra/outputs/explore_eqa_gpt-4o-2024-08-06_heirar_view/'
    filepath1 = '/home/saumyas/catkin_ws_semnav/src/hydra/outputs/explore_eqa_gpt-4o-2024-08-06_heirar_view_traj_len/'

    outfile = filepath1 + 'metrics_succ.json'
    with open(filepath1+'gpt-4o-2024-08-06_images_True.json', 'r') as file:
        data1 = json.load(file)

    with open(filepath2+'gpt-4o-2024-08-06_images_True.json', 'r') as file:
        data2 = json.load(file)
    
    num_success, planning_steps_all_trajs, length_all_trajs = 0, 0, 0
    total_trajs = len(data1.keys())

    for k, v in data1.items():
        if v['Success']:
            num_success += 1
            length_all_trajs += v['metrics']['traj_length']
            planning_steps_all_trajs += v['metrics']['vlm_steps']

    for k, v in data2.items():
        if 'traj_length' in v['metrics'].keys():
            total_trajs += 1
            if v['Success']:
                num_success += 1
                length_all_trajs += v['metrics']['traj_length']
                planning_steps_all_trajs += v['metrics']['vlm steps']

    metrics = {}
    metrics['length_all_trajs'] = length_all_trajs
    metrics['planning_steps_all_trajs'] = float(planning_steps_all_trajs)
    metrics['num_success'] = float(num_success)
    metrics['total_trajs'] = total_trajs

    print(f"Saving file: {outfile}")
    with open(outfile, 'w') as file:
        json.dump(metrics, file, indent=4)
    print(f"Saved file: {outfile}")
