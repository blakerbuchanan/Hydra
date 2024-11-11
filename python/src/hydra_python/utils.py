import hydra_python as hydra
import csv, os, ast
import click
import numpy as np

def get_latest_image(output_folder):
    png_files = [file for file in os.listdir(output_folder) if file.startswith('current_img_')]
    indices = [int(f.split('current_img_')[1].split('.png')[0]) for f in png_files]
    return output_folder / f"current_img_{np.max(indices)}.png"

def hydra_get_mesh(pipeline):
    vertices = pipeline.graph.mesh.get_vertices()
    faces = pipeline.graph.mesh.get_faces()

    mesh_vertices = vertices[:3, :].T
    mesh_triangles = faces.T
    mesh_colors = vertices[3:, :].T

    return mesh_vertices, mesh_colors, mesh_triangles
    
def project_2d_to_3d(bbox_2d, depth_image, intrinsics, extrinsics):
    # Get the 2D bounding box corners
    x_min, y_min, x_max, y_max = bbox_2d

    # Get the pixel locations of the corners of the bounding box
    bbox_corners_2d = np.array([
        [x_min, y_min],  # top-left
        [x_max, y_min],  # top-right
        [x_max, y_max],  # bottom-right
        [x_min, y_max]   # bottom-left
    ])

    # Retrieve the depth at each corner of the bounding box
    depth_corners = np.array([depth_image[y, x] for x, y in bbox_corners_2d])

    # Create a 3D point cloud for each corner in camera space
    # Using the intrinsic matrix to back-project the points
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Convert the 2D points into normalized image coordinates and back-project to 3D
    points_3d_camera = []
    for i, (x, y) in enumerate(bbox_corners_2d):
        Z = depth_corners[i]
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        points_3d_camera.append([X, Y, Z])

    points_3d_camera = np.array(points_3d_camera)

    # Convert points from camera space to world space using the extrinsic matrix
    points_3d_camera_homogeneous = np.hstack([points_3d_camera, np.ones((points_3d_camera.shape[0], 1))])  # Convert to homogeneous coordinates
    points_3d_world = np.dot(extrinsics, points_3d_camera_homogeneous.T).T[:, :3]  # Drop the homogeneous coordinate (4th)

    return points_3d_world

import open3d as o3d
def get_bb_from_sem(habitat_data):
    label_names = {i: x for i, x in enumerate(habitat_data.colormap.names)}

    pose_cam = get_cam_pose_tsdf(habitat_data.get_depth_sensor_state())
    world_t_body = pose_cam[:3, 3]
    q_xyzw = R.from_matrix(pose_cam[:3, :3]).as_quat()
    q_wxyz = np.roll(q_xyzw, 1)

    # Create point cloud from RGB-D image
    depth_image = o3d.geometry.Image(habitat_data.depth.astype(np.float32))
    rgb_image = o3d.geometry.Image(np.ascontiguousarray(habitat_data.rgb))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, depth_scale=1.0, convert_rgb_to_intensity=False)

    camera_info = habitat_data.camera_info
    width, height, fx, fy, cx, cy = camera_info['width'], camera_info['height'], camera_info['fx'], camera_info['fy'], camera_info['cx'], camera_info['cy']

    # Define camera intrinsic matrix
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    # Create a point cloud from the RGB-D image
    pcd_camera = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, np.linalg.inv(pose_cam))

    # Get points and corresponding colors from point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    points_camera = np.asarray(pcd_camera.points)
    colors_camera = np.asarray(pcd_camera.colors)

    # Get the pixel coordinates of the valid depth points in the 2D image plane
    u = np.clip(np.round(points_camera[:, 0] / points_camera[:, 2] * fx + cx).astype(int), 0, width - 1)
    v = np.clip(np.round(points_camera[:, 1] / points_camera[:, 2] * fy + cy).astype(int), 0, height - 1)

    # Extract mask values at the corresponding 2D pixel coordinates
    mask_values = habitat_data.labels[v, u]

    # Mask the points based on the semantic mask
    object_labels = np.unique(habitat_data.labels)
    for object_label in object_labels:

        valid_points = points[(mask_values == object_label)]
        valid_colors = colors[(mask_values == object_label)]

        # Create a new point cloud with only valid points
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(valid_points)
        pcd_filtered.colors = o3d.utility.Vector3dVector(valid_colors)

        # Extract a 3D bounding box from the point cloud
        bbox = pcd_filtered.get_axis_aligned_bounding_box()
        rr.log(f"world/debug_{label_names[object_label]}", rr.Points3D(bbox.max_bound, colors=[255, 0, 0], radii=0.11))
    return bbox


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

def get_instruction_from_eqa_data(question_data):
    question = question_data["question"]
    # self.choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
    clean_ques_ans = question_data["question"]
    choices = ast.literal_eval(question_data["choices"])
    # Re-format the question to follow LLaMA style
    vlm_question = question
    vlm_pred_candidates = ["A", "B", "C", "D"]
    for token, choice in zip(vlm_pred_candidates, choices):
        vlm_question += "\n" + token + "." + " " + choice
        if ("do not choose" not in choice.lower()) and (choice.lower() not in ['yes', 'no']):
            clean_ques_ans += "  " + token + "." + " " + choice
    return vlm_question, clean_ques_ans, choices, vlm_pred_candidates

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

from omegaconf import OmegaConf
from pathlib import Path
def initialize_hydra_pipeline_rosbag(cfg, camera_info, output_path):
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

    config_path = Path(__file__).resolve().parent.parent.parent.parent / 'config/label_spaces/hm3d_label_space.yaml'
    hm3d_labelspace = OmegaConf.load(config_path)

    names = [d.name for d in hm3d_labelspace.label_names]
    colormap = hydra.SegmentationColormap.from_names(names=names)
    pipeline_config.label_names = {i: x for i, x in enumerate(colormap.names)}
    colormap.fill_label_space(pipeline_config.label_space) # TODO: check
    
    # pipeline_config.label_space.colormap = {0: (np.array([255,255,255])).astype(np.uint8).tolist()}
    if output_path:
        pipeline_config.logs.log_dir = str(output_path)
    pipeline = hydra.HydraPipeline(
        pipeline_config, robot_id=0, config_verbosity=cfg.config_verbosity, freeze_global_info=False)
    pipeline.init(configs, hydra.create_camera(camera_info))

    if output_path:
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir()
        hydra.set_glog_dir(str(glog_dir))
    
    return pipeline

def initialize_hydra_pipeline_stretch(cfg, obs, output_path, sensor_categories_mapping=None):

    # Get camera info
    camera_K = obs.camera_K
    width = obs.rgb.shape[1]
    height = obs.rgb.shape[0]
    camera_info = {
        "fx": float(camera_K[0,0]),
        "fy": float(camera_K[1,1]),
        "cx": float(camera_K[0,2]),
        "cy": float(camera_K[1,2]),
        "width": width,
        "height": height,
    }

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

    if sensor_categories_mapping is not None:
        names = [v for k, v in sensor_categories_mapping.items()]
    else:
        click.secho(f"Using default label space from habitat'",fg="red",)
        config_path = Path(__file__).resolve().parent.parent.parent.parent / 'config/label_spaces/hm3d_label_space.yaml'
        hm3d_labelspace = OmegaConf.load(config_path)
        names = [d.name for d in hm3d_labelspace.label_names]

    colormap = hydra.SegmentationColormap.from_names(names=names)
    pipeline_config.label_names = {i: x for i, x in enumerate(colormap.names)}
    colormap.fill_label_space(pipeline_config.label_space) # TODO: check
    
    if output_path:
        pipeline_config.logs.log_dir = str(output_path)
    pipeline = hydra.HydraPipeline(
        pipeline_config, robot_id=0, config_verbosity=cfg.config_verbosity, freeze_global_info=False)
    pipeline.init(configs, hydra.create_camera(camera_info))

    if output_path:
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir()
        hydra.set_glog_dir(str(glog_dir))
    
    return pipeline

def get_traj_len_from_poses(poses):
    pts = np.array([pt[1] for pt in poses])
    deltas = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    return np.sum(segment_lengths)
    
if __name__ == "__main__":
    get_latest_image(Path("/home/saumyas/catkin_ws_semnav/src/hydra/outputs/test_obj_enrich/0_00006-HkseAnWCgqk_0"))