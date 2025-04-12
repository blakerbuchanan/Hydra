import numpy as np
import json

def main():
    question_path = '/mnt/hdd1/saumyas/data/semnav/scanQA/ScanQA_v1.0_train.json'
    scannet_data_path = '/mnt/hdd1/saumyas/data/semnav/scannet/scans'
    with open(question_path, 'r') as file:
        questions_data = json.load(file)
    
    # len(questions_data) = 25563
    scannet_scans, scannet_spaces, apartment_scenes = [], [], []
    for i, data in enumerate(questions_data):
        scan_id = data['scene_id'].split('scene')[1]
        space_id = scan_id.split('_')[0]
        scannet_scans.append(scan_id)
        scannet_spaces.append(space_id)

        # Checking the number of apartment scenes
        txt_file = f"{scannet_data_path}/{data['scene_id']}/{data['scene_id']}.txt"
        with open(txt_file, "r") as f:
            for line in f:
                if line.startswith("sceneType"):
                    sceneType = line.split("=")[1].strip()
                    if sceneType.lower() == "apartment":
                        apartment_scenes.append(space_id)
                        print(f"Q {i}: {data['question']}  A: {data['answers']}")
    
    unique_scans = np.unique(scannet_scans)
    unique_spaces = np.unique(scannet_spaces)
    unique_apartment_scenes = np.unique(apartment_scenes)
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main()