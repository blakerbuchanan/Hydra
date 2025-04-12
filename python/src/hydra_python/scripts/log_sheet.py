import gspread
import pickle, time
from tqdm import trange
import json
# https://docs.gspread.org/en/latest/user-guide.html

if __name__ == "__main__":
    gc = gspread.service_account('/home/saumyas/.config/gspread/saumya-default-project-8b699ae84be9.json')
    sh = gc.open("GraphEQA results 2024")
    # worksheet = sh.worksheet("Analysis zeroshot")
    worksheet = sh.worksheet("OpenEQA")
    # print(worksheet.acell('A1').value)

    method_col = worksheet.find('GraphEQA', in_row=1).col
    
    results_path = '/home/saumyas/catkin_ws_semnav/src/hydra/outputs/openEQA_gpt-4o-2024-08-06_grapheqa_single_floor_with_choices/gpt-4o-2024-08-06_images_True.json'
    with open(results_path, 'r') as file:
        results = json.load(file)

    for k, v in results.items():
        ind = k.split('_')[0]
        
        log_succ = False
        while not log_succ:
            try:
                q_row = worksheet.find(str(ind), in_column=1).row
                if v["Success"]:
                    worksheet.update_cell(q_row, method_col, 1)
                else:
                    worksheet.update_cell(q_row, method_col, 0)
                log_succ = True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 60")
                time.sleep(60)