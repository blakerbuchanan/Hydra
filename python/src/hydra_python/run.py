"""Module containing code for running Hydra."""
from scipy.spatial.transform import Rotation as R
import numpy as np
import click
import rerun as rr
import rerun.blueprint as rrb

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


def hydra_output_callback(pipeline, visualizer):
    """Show graph."""
    if visualizer:
        visualizer.update_graph(pipeline.graph)


def _take_step(pipeline, data, pose, segmenter, image_viz):
    timestamp, world_t_body, q_wxyz = pose
    q_xyzw = np.roll(q_wxyz, -1)

    world_T_body = np.eye(4)
    world_T_body[:3, 3] = world_t_body
    world_T_body[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    data.set_pose(timestamp, world_T_body)

    labels = segmenter(data.rgb) if segmenter else data.labels
    if image_viz:
        image_viz.show(data.colormap(labels))

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
):
    """Do stuff."""
    image_viz = ImageVisualizer() if show_images else None

    # Initialize Rerun and specify the .rrd file for logging
    full_output_path = "habitat_data.rrd"
    rr.init("example_rgb_image_logging")
    rr.save(full_output_path)

    primary_camera_entity = "world/camera_lowres"

    # Define a blueprint with an image space for logging the RGB image data
    blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial3DView(name="3D"),
            rrb.TextDocumentView(name="PlannerOutput"),
            ),
        # Note that we re-project the annotations into the 2D views:
        # For this to work, the origin of the 2D views has to be a pinhole camera,
        # this way the viewer knows how to project the 3D annotations into the 2D views.
        rrb.Vertical(
            rrb.Spatial2DView(
                name="RGB",
                origin=primary_camera_entity,
                contents=["$origin/rgb", "/world/annotations/**"],
            ),
            rrb.Spatial2DView(
                    name="Semantic Labels",
                    origin=primary_camera_entity,
                    contents=["$origin/semantic", "/world/annotations/**"],
            ),
            )
    )

    rr.send_blueprint(blueprint)

    imgs_colormap, imgs_rgb, imgs_labels = [], [], []

    agent_positions = []
    if show_progress:
        with click.progressbar(pose_source) as bar:
            for pose in bar:
                # We can change this directory when we determine how we want to save out the gifs at the end
                pipeline.graph.save(output_path / "dsg.json", False)
                pipeline.graph.save_filtered(output_path / "filtered_dsg.json", False)
                _take_step(pipeline, data, pose, segmenter, image_viz)
                if step_callback:
                    step_callback(pipeline, visualizer)
    else:
        for pose in pose_source:
            pipeline.graph.save(output_path / "dsg.json", False)
            pipeline.graph.save_filtered(output_path / "filtered_dsg.json", False)

            _take_step(pipeline, data, pose, segmenter, image_viz)
            imgs_colormap.append(data.colormap(data.labels))
            imgs_labels.append(data.labels)
            imgs_rgb.append(data.rgb)

            # BEGIN RERUN LOGGING
            # log the camera transform, rgb image, and depth image
            # rr.log("world/camera_lowres", rr.Transform3D(transform=camera_from_world))
            # rr.log("world/camera_lowres", rr.Pinhole(image_from_camera=intrinsic, resolution=[w, h]))
            rr.log(f"{primary_camera_entity}/rgb", rr.Image(data.rgb).compress(jpeg_quality=95))
            rr.log(f"{primary_camera_entity}/semantic", rr.Image(data.colormap(data.labels)).compress(jpeg_quality=95))

            agent_positions.append(data.get_state())

            # log the agent trajectory
            rr.log("world/trajectory", rr.LineStrips3D(agent_positions, colors=[0, 255, 0]))

            vertices = pipeline.graph.mesh.get_vertices()
            faces = pipeline.graph.mesh.get_faces()

            mesh_vertices = vertices[:3, :].T
            mesh_triangles = faces.T
            mesh_colors = vertices[3:, :].T

            # log the mesh data
            vp, vc, ti = mesh_vertices, mesh_colors, mesh_triangles
            rr.log(
                "world/mesh",
                rr.Mesh3D(
                    vertex_positions=vp,
                    vertex_colors=vc,
                    triangle_indices=ti,
                ),
                timeless=False,
            )

            # log the text to the Planner Output window
            rr.log(
                "PlannerOutput",
                rr.TextDocument(
                    "this is the planner output",
                    media_type=rr.MediaType.TEXT,
                ),
            )

            ## ***Process pipeline.graph***
            place_node_positions = []
            active_frontier_place_node_positions = []
            object_node_positions = []
            room_node_positions = []
            building_node_positions = []
            n_object_nodes, n_place_nodes, n_frontier_nodes, n_agent_nodes, n_room_nodes, n_building_nodes = 0, 0, 0, 0, 0, 0

            for node in pipeline.graph.nodes:
                if 'p' in node.id.category.lower():
                    # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}. Place: {node.attributes.active_frontier}")
                    place_node_positions.append(node.attributes.position)
                    n_place_nodes += 1
                if 'f' in node.id.category.lower():
                    # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}. Active Frontier: {node.attributes.active_frontier}")
                    active_frontier_place_node_positions.append(node.attributes.position)
                    n_frontier_nodes += 1
                if 'o' in node.id.category.lower():
                    # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}.")
                    object_node_positions.append(node.attributes.position)
                    n_object_nodes += 1

                    # log the bounding boxes
                    bbox = node.attributes.bounding_box

                    half_size = 0.5 * np.array(bbox.dimensions).reshape(-1, 3)[0]
                    centroid = np.array(bbox.world_P_center).reshape(-1, 3)[0]
                    mat3x3 = np.array(bbox.world_R_center).reshape(3, 3)
                    label = node.attributes.name

                    rr.log(
                        f"world/annotations/box-{label}",
                        rr.Boxes3D(
                            half_sizes=half_size,
                            centers=centroid,
                            labels=label,
                        ),
                        rr.InstancePoses3D(mat3x3=mat3x3),
                        timeless=False,
                    )

                if 'r' in node.id.category.lower():
                    # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}.")
                    room_node_positions.append(node.attributes.position)
                    n_room_nodes += 1
                if 'b' in node.id.category.lower():
                    # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}.")
                    building_node_positions.append(node.attributes.position)
                    n_building_nodes += 1

            place_node_positions = np.array(place_node_positions)
            active_frontier_place_node_positions = np.array(active_frontier_place_node_positions)
            object_node_positions = np.array(object_node_positions)
            room_node_positions = np.array(room_node_positions)
            building_node_positions = np.array(building_node_positions)

            # log the frontier nodes with color red
            rr.log(
                "frontier_nodes",
                rr.Points3D(active_frontier_place_node_positions, colors=[255, 0, 0], radii=0.08)
            )

            # END RERUN LOGGING

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
