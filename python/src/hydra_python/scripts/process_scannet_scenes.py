import numpy as np
import json
import tqdm
from pathlib import Path
import itertools

def main():
    room_labels = {"kitchen", "living room", "bathroom", "bedroom", "hallway", "office"}

    # scannet_paths = ['/mnt/hdd1/saumyas/data/semnav/scannet/scans', '/mnt/hdd1/saumyas/data/semnav/scannet/scans_test']
    scannet_paths = ['/mnt/hdd1/saumyas/data/semnav/scannet/scans']
    folders = [sorted(Path(scannet_path).glob("*")) for scannet_path in scannet_paths]
    
    unique_labels, apartment_scenes = [], []
    for folder in itertools.chain(*folders):
        # file = sorted(folder.glob("*_vh_clean.aggregation.json"))
        # if len(file) > 0:
        #     with open(file[0], 'r') as f:
        #         agg_data = json.load(f)
            
        #     scene_categories = [seg["label"] for seg in agg_data["segGroups"]]
        #     unique_labels.extend(np.unique(scene_categories))

        #     rooms_in_scan = {seg["label"] for seg in agg_data["segGroups"]}.intersection(room_labels)
            
        #     multi_room = len(rooms_in_scan) > 1
        #     print(f'{rooms_in_scan=} {multi_room=}')
        scan_id = str(folder).split('scene')[1]
        space_id = scan_id.split('_')[0]
        txt_file = sorted(folder.glob("*.txt"))
        if len(txt_file) > 0:
            with open(txt_file[0], "r") as f:
                for line in f:
                    if line.startswith("sceneType"):
                        sceneType = line.split("=")[1].strip()  # Extract value after "="
                        if sceneType.lower() == "apartment":
                            apartment_scenes.append(space_id)
    
    unique_labels_all = np.unique(unique_labels)
    unique_apartment_scenes = np.unique(apartment_scenes)
    import ipdb; ipdb.set_trace()
    
if __name__ == "__main__":
    main()