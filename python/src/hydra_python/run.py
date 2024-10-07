"""Module containing code for running Hydra."""
from scipy.spatial.transform import Rotation as R
import numpy as np
import click
import rerun as rr
import rerun.blueprint as rrb
from hydra_python.frontier_mapping_eqa.utils import get_cam_pose_tsdf, pos_habitat_to_normal
from voxel_mapping import Observations
import threading, time

class ImageVisualizer:
    """GUI for showing images."""

    def __init__(self):
        """Initialize pyqtgraph."""
        # TODO(nathan) make a grid layout
        import pyqtgraph.multiprocess as mp

        proc = mp.QtProcess(processRequests=False)
        self._pg = proc._import("pyqtgraph")
        self._pg.setConfigOptions(imageAxisOrder="row-major")
        self._view = None

    def show(self, image, is_depth=False):
        """Show an image."""
        kwargs = {"autoLevels": False, "levels": (0, 255)} if not is_depth else {}
        if self._view is None:
            self._view = self._pg.image(image, **kwargs)
            if is_depth:
                self._view.setPredefinedGradient("viridis")
        else:
            self._view.setImage(image, **kwargs)

def hydra_get_mesh(pipeline):
    vertices = pipeline.graph.mesh.get_vertices()
    faces = pipeline.graph.mesh.get_faces()

    mesh_vertices = vertices[:3, :].T
    mesh_triangles = faces.T
    mesh_colors = vertices[3:, :].T

    return mesh_vertices, mesh_colors, mesh_triangles

def hydra_get_object_place_nodes(pipeline):

    place_node_positions = []
    frontier_node_positions = []
    room_node_positions = []
    building_node_positions = []

    object_node_positions, bb_half_sizes, bb_centroids, bb_mat3x3, bb_labels, bb_colors = [], [], [], [], [], []

    for node in pipeline.graph.nodes:
        if 'p' in node.id.category.lower():
            place_node_positions.append(node.attributes.position)
        if 'f' in node.id.category.lower():
            frontier_node_positions.append(node.attributes.position)
        if 'o' in node.id.category.lower():
            object_node_positions.append(node.attributes.position)

            # log the bounding boxes
            bbox = node.attributes.bounding_box

            bb_half_sizes.append(0.5 * bbox.dimensions)
            bb_centroids.append(bbox.world_P_center)
            bb_mat3x3.append(bbox.world_R_center)
            bb_labels.append(node.attributes.name)
            bb_colors.append(node.attributes.color)

        if 'r' in node.id.category.lower():
            room_node_positions.append(node.attributes.position)
        if 'b' in node.id.category.lower():
            building_node_positions.append(node.attributes.position)

    node_info = {
        'place_node_positions': place_node_positions,
        'frontier_node_positions': frontier_node_positions,
        'room_node_positions': room_node_positions,
        'building_node_positions': building_node_positions,
        'object_node_info': {
            'object_node_positions': object_node_positions,
            'bb_half_sizes': bb_half_sizes,
            'bb_centroids': bb_centroids,
            'bb_mat3x3': bb_mat3x3,
            'bb_labels': bb_labels,
            'bb_colors': bb_colors,
        }

    }
    return node_info

# def get_in_plane_frontier_nodes(frontier_node_positions, agent_pos):
#     if len(frontier_node_positions)>0:
#         inplace_idxs = is_relevant_frontier(np.array(frontier_node_positions), agent_pos)
#         inplane_frontier_node_positions = np.array(frontier_node_positions)[inplace_idxs]
#         return inplane_frontier_node_positions
#     else:
#         return []


def hydra_output_callback(pipeline, visualizer):
    """Show graph."""
    if visualizer:
        visualizer.update_graph(pipeline.graph)

def _take_step(pipeline, data, pose, segmenter, image_viz, is_eqa=False):
    timestamp, world_t_body, q_wxyz = pose
    q_xyzw = np.roll(q_wxyz, -1) #changing to xyzw format

    world_T_body = np.eye(4)
    world_T_body[:3, 3] = world_t_body
    world_T_body[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    data.set_pose(timestamp, world_T_body, is_eqa=is_eqa)

    labels = segmenter(data.rgb) if segmenter else data.labels
    if image_viz:
        image_viz.show(data.colormap(labels))

    if is_eqa:
        pose_cam = get_cam_pose_tsdf(data.get_depth_sensor_state())
        world_t_body = pose_cam[:3, 3]
        q_xyzw = R.from_matrix(pose_cam[:3, :3]).as_quat()
        q_wxyz = np.roll(q_xyzw, 1)

    pipeline.step(timestamp, world_t_body, q_wxyz, data.depth, labels, data.rgb)

import imageio, cv2
def run(
    pipeline,
    data,
    pose_source,
    segmenter=None,
    visualizer=None,
    show_images=False,
    show_progress=True,
    step_callback=hydra_output_callback,
    output_path=None,
    suffix=' ',
    rr_logger=None,
    vlm_planner=None,
    is_eqa=False,
):
    """Do stuff."""
    image_viz = ImageVisualizer() if show_images else None

    imgs_colormap, imgs_rgb, imgs_labels = [], [], []

    agent_positions, agent_quats_wxyz = [], []
    if show_progress:
        with click.progressbar(pose_source) as bar:
            for pose in bar:
                # We can change this directory when we determine how we want to save out the gifs at the end
                pipeline.graph.save(output_path / "dsg.json", False)
                pipeline.graph.save_filtered(output_path / "filtered_dsg.json", False)
                _take_step(pipeline, data, pose, segmenter, image_viz, is_eqa=is_eqa)
                if step_callback:
                    step_callback(pipeline, visualizer)
    else:
        for pose in pose_source:
            pipeline.graph.save(output_path / "dsg.json", False)
            pipeline.graph.save_filtered(output_path / "filtered_dsg.json", False)

            _take_step(pipeline, data, pose, segmenter, image_viz, is_eqa=is_eqa)
            imgs_colormap.append(data.colormap(data.labels))
            imgs_labels.append(data.labels)
            imgs_rgb.append(data.rgb)

            agent_pos, agent_quat_wxyz = data.get_state(is_eqa=is_eqa)
            agent_positions.append(agent_pos)
            agent_quats_wxyz.append(agent_quat_wxyz)

            camera_pos, camera_quat_wxyz = data.get_camera_pos(is_eqa=is_eqa)
            mesh_vertices, mesh_colors, mesh_triangles = hydra_get_mesh(pipeline)
            # node_info = hydra_get_object_place_nodes(pipeline)
            # inplane_frontier_node_positions = get_in_plane_frontier_nodes(node_info['frontier_node_positions'], agent_positions[-1])
            # inplane_place_node_positions = get_in_plane_frontier_nodes(node_info['place_node_positions'], agent_positions[-1])
            if vlm_planner is not None:
                vlm_planner.sg_sim.update()
            if rr_logger is not None:
                rr_logger.log_mesh_data(mesh_vertices, mesh_colors, mesh_triangles)
                rr_logger.log_agent_data(agent_positions)
                rr_logger.log_agent_tf(agent_pos, agent_quat_wxyz)
                rr_logger.log_camera_tf(camera_pos, camera_quat_wxyz)
                rr_logger.log_img_data(data)
                rr_logger.step()

            if step_callback:
                step_callback(pipeline, visualizer)
            
    # Parameters for text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0)
    thickness = 1
    step_size = 2000  # Adjust step size for sparse labeling
    labeled_frames = []
    for idx in range(len(imgs_colormap)):
        color_img = imgs_colormap[idx].copy()
        unique_labels = np.unique(imgs_labels[idx])
        for label in unique_labels:
            points = np.argwhere(imgs_labels[idx] == label)
            for i, point in enumerate(points[::step_size]):
                y, x = point
                cv2.putText(color_img, str(f'{label}:{data.colormap.names[label]}'), (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
                #rr.log(f"{primary_camera_entity}/rgb/label", rr.TextLog(f"{label}", position=[x, y]))

        labeled_frames.append(color_img)

    imageio.mimsave(output_path / f'images_hm3d_semantic_{suffix}.gif', imgs_colormap)
    imageio.mimsave(output_path / f'images_hm3d_rgb_{suffix}.gif', imgs_rgb)
    imageio.mimsave(output_path / f'images_hm3d_labeled_frame_{suffix}s.gif', labeled_frames)

    #rr.shutdown()

def run_eqa(
    pipeline,
    habitat_data,
    pose_source,
    segmenter=None,
    step_callback=hydra_output_callback,
    output_path=None,
    rr_logger=None,
    vlm_planner=None,
    tsdf_planner=None,
    voxel_space=None,
):

    agent_positions, agent_quats_wxyz = [], []
    step_time = frontier_update_time = voxel_log_time = sg_update_time = mesh_log_time = 0
    for pose in pose_source:
        pipeline.graph.save(output_path / "dsg.json", False)
        pipeline.graph.save_filtered(output_path / "filtered_dsg.json", False)

        start = time.time()
        _take_step(pipeline, habitat_data, pose, segmenter, image_viz=None, is_eqa=True)
        step_time += time.time()-start

        agent_pos, agent_quat_wxyz = habitat_data.get_state(is_eqa=True)
        agent_positions.append(agent_pos)
        agent_quats_wxyz.append(agent_quat_wxyz)
        camera_pos, camera_quat_wxyz = habitat_data.get_camera_pos(is_eqa=True)
        mesh_vertices, mesh_colors, mesh_triangles = hydra_get_mesh(pipeline)

        cam_pose_tsdf = get_cam_pose_tsdf(habitat_data.get_depth_sensor_state())
        pts_normal = pos_habitat_to_normal(pose[1])

        if tsdf_planner:
            tsdf_planner.update(
                habitat_data.rgb,
                habitat_data.depth,
                pts_normal,
                cam_pose_tsdf,
            )
            frontier_nodes = tsdf_planner.frontier_to_sample_normal
            
        if voxel_space:
            start = time.time()
            obs = Observations(
                gps=pts_normal[:2],
                compass=habitat_data.get_heading_angle(),
                camera_pose=cam_pose_tsdf,
                rgb=habitat_data.rgb,
                depth=habitat_data.depth,
                xyz=None,
                camera_K=voxel_space.cam_intr,
            )
            voxel_space.voxel_map.add_obs(obs)
            frontier_update_time += time.time()-start

            # start = time.time()
            # voxel_space.update()
            # voxel_log_time += time.time()-start

            # frontier_nodes = voxel_space.outside_frontier_points

        # if vlm_planner:
        #     start = time.time()
        #     vlm_planner.sg_sim.update(frontier_nodes)
        #     sg_update_time += time.time()-start

        if rr_logger:
            start = time.time()
            rr_logger.log_mesh_data(mesh_vertices, mesh_colors, mesh_triangles)
            rr_logger.log_agent_data(agent_positions)
            rr_logger.log_agent_tf(agent_pos, agent_quat_wxyz)
            rr_logger.log_camera_tf(camera_pos, camera_quat_wxyz)
            rr_logger.log_img_data(habitat_data)
            mesh_log_time += time.time()-start
            # if voxel_space:
            #     rr_logger.log_clear("world/voxel")
                # rr_logger.log_voxel_map(voxel_space)
            rr_logger.step()

        if step_callback:
            step_callback(pipeline, None)
    if voxel_space:   
        start = time.time()
        voxel_space.update()
        voxel_log_time += time.time()-start
        frontier_nodes = voxel_space.outside_frontier_points
    if vlm_planner:
        start = time.time()
        vlm_planner.sg_sim.update(frontier_nodes)
        sg_update_time += time.time()-start
    print(f"{step_time=} {frontier_update_time=} {voxel_log_time=} {sg_update_time=} {mesh_log_time=}")