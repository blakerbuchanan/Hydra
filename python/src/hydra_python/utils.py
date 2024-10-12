import hydra_python as hydra
import csv, os
import click

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
        pipeline_config, robot_id=0, config_verbosity=cfg.config_verbosity, freeze_global_info=False)
    pipeline.init(configs, hydra.create_camera(habitat_data.camera_info))

    if output_path:
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir()
        hydra.set_glog_dir(str(glog_dir))
    
    return pipeline