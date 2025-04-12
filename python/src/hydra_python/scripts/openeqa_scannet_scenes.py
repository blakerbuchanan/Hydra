import numpy as np
import json

def main():
    question_path = '/home/saumyas/catkin_ws_semnav/data/openEQA/open-eqa-v0.json'
    with open(question_path, 'r') as file:
        questions_data = json.load(file)
    
    scannet_scans, scannet_spaces = [], []
    for data in questions_data:
        if 'scannet' in data['episode_history']:
            scan_id = data['episode_history'].split('scene')[1]
            space_id = scan_id.split('_')[0]
            scannet_scans.append(scan_id)
            scannet_spaces.append(space_id)
    
    unique_scans = np.unique(scannet_scans)
    unique_spaces = np.unique(scannet_spaces)
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main()