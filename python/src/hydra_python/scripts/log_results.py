import pickle, json
import numpy as np
import csv, os, time
import gspread
from tqdm import trange, tqdm

def load_eqa_data():
    # Load dataset
    with open('/home/saumyas/Projects/semnav/explore-eqa_semnav/data/questions.csv') as f:
        full_questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    # Filter to include only scenes with semantic annotations
    semantic_annot_data_path = '/home/saumyas/catkin_ws_semnav/data/hm3d-train-semantic-annots-v0.2'
    semantic_scenes = [f for f in os.listdir(semantic_annot_data_path) if os.path.isdir(os.path.join(semantic_annot_data_path, f))]

    questions_data = []
    for data in full_questions_data:
        if data['scene'] in semantic_scenes:
            questions_data.append(data)

    return questions_data

def load_spreadsheet(method='Explore-EQA'):
    gc = gspread.service_account('/home/saumyas/.config/gspread/saumya-default-project-8b699ae84be9.json')
    sh = gc.open("GraphEQA results 2024")
    worksheet = sh.worksheet("Sheet4")
    method_col = worksheet.find(method, in_row=1).col
    return worksheet, method_col

def get_results(method='GraphEQA'):
    filepath = '/home/saumyas/catkin_ws_semnav/src/hydra/outputs/explore_eqa_gpt-4o-2024-08-06_heirar_view/'
    with open(filepath+'gpt-4o-2024-08-06_images_True_new.json', 'r') as file:
        data = json.load(file)
    return data

def load_idx_file():
    with open('/home/saumyas/Projects/semnav/explore-eqa_semnav/results/sem_to_all_indx.json', 'r') as file:
        sem_to_all_indx = json.load(file)
    semantic_all_idxs = [v['all_idx'] for k,v in sem_to_all_indx.items()]
    return sem_to_all_indx, semantic_all_idxs

if __name__ == "__main__":

    method = 'GraphEQA_steps'
    questions_data = load_eqa_data()
    sem_to_all_indx, semantic_all_idxs = load_idx_file()
    worksheet, method_col = load_spreadsheet(method)
    results = get_results()

    for question_ind, question_data in enumerate(tqdm(questions_data)):
        experiment_id = f'{question_ind}_{question_data["scene"]}_{question_data["floor"]}'
        if experiment_id in results.keys():
            
            log_succ = False
            while not log_succ:
                try:   
                    all_idx = sem_to_all_indx[str(question_ind)]['all_idx']
                    q_row = worksheet.find(str(all_idx), in_column=1).row

                    # if results[experiment_id]['Success']:
                    #     worksheet.update_cell(q_row, method_col, 1)
                    # else:
                    #     worksheet.update_cell(q_row, method_col, 0)

                    worksheet.update_cell(q_row, method_col, results[experiment_id]['metrics']['vlm_steps'])
                    log_succ = True
                except Exception as e:
                    print(f"An error occurred: {e}. Sleeping for 60")
                    time.sleep(60)
