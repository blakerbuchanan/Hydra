import rerun as rr
import rerun.blueprint as rrb
import numpy as np

class RRLogger:
    def __init__(self, output_path):
        # Initialize Rerun and specify the .rrd file for logging
        full_output_path = output_path / "test_logger.rrd"

        self._timeline = "vlm_plan_logging"
        rr.init(self._timeline)
        rr.save(full_output_path)

        self.primary_agent_entity = "world/agent"

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
                    origin=self.primary_agent_entity,
                    contents=["$origin/rgb", "/world/annotations/**"],
                ),
                rrb.Spatial2DView(
                        name="Semantic Labels",
                        origin=self.primary_agent_entity,
                        contents=["$origin/semantic", "/world/annotations/**"],
                ),
                )
        )

        rr.send_blueprint(blueprint)

        self._node_color_map = {
            'object': [225,225,0],
            'frontier': [255,0,0],
            'visited': [0,0,0],
            'room': [255,0,255],
            'building': [0,255,255],
            'agent': [0,0,255],
        }
        self._edge_color_map = {
            'building-to-room': [225,0,0],
            'room-to-visited': [0,255,0],
            'visited-to-object': [0,0,255],
            'visited-to-frontier': [255,255,255],
            'visited-to-visited': [0,0,0],
            'visited-to-agent': [255,255,0],
        }
        self.reset()

    def reset(self):
        self._t = 0
        self._dt = 0.1
        rr.set_time_seconds(self._timeline, self._t)

    def log_mesh_data(self, mesh_vertices, mesh_colors, mesh_triangles):
        
        rr.log(
            "world/mesh",
            rr.Mesh3D(
                vertex_positions=mesh_vertices,
                vertex_colors=mesh_colors,
                triangle_indices=mesh_triangles,
            ),
            timeless=False,
        )
    
    def log_agent_data(self, agent_positions):
        rr.log(f"world/robot_traj", rr.LineStrips3D(agent_positions, colors=[0, 0, 255]))
        rr.log(f"world/robot_pos", rr.Points3D(agent_positions[-1], colors=[0, 0, 255], radii=0.11))

    def log_traj_data(self, agent_positions):
        rr.log("world/desired_traj", rr.LineStrips3D(agent_positions, colors=[0, 255, 255]))
    
    def log_agent_tf(self, pos, quat):
        translation = np.asarray([pos[0], pos[1], pos[2]])
        quat_mod = np.asarray([quat[1], quat[2], quat[3], quat[0]])
        agent_from_world = rr.Transform3D(
            translation=translation, rotation=rr.Quaternion(xyzw=quat_mod), from_parent=False
        )
        rr.log(f"{self.primary_agent_entity}", agent_from_world)

    def log_target_poses(self, target_poses):
        rr.log("world/target_poses", rr.Points3D(target_poses, colors=[0,255,0], radii=0.11))

    def log_text_data(self, text):
        rr.log(
            "PlannerOutput",
            rr.TextDocument(
                text,
                media_type=rr.MediaType.TEXT,
            ),
        )

    def log_navmesh_data(self, navmesh):
        # log the frontier nodes with color red
        rr.log(
            f"world/navmesh_nodes",
            rr.Points3D(navmesh, colors=[255,255,255], radii=0.11)
        )

    def log_frontier_data(self, frontier_node_positions):
        # log the frontier nodes with color red
        rr.log(
            "world/frontier_nodes",
            rr.Points3D(frontier_node_positions, colors=[255,0,0], radii=0.08)
        )

    def log_selected_frontier_data(self, frontier_node_positions):
        # log the frontier nodes with color red
        rr.log(
            "world/selected_frontier_nodes",
            rr.Points3D(frontier_node_positions, colors=[255, 255, 0], radii=0.08)
        )
    
    def log_place_data(self, place_node_positions):
        # log the place nodes with color red
        rr.log(
            "world/place_nodes",
            rr.Points3D(place_node_positions, colors=[255,255,255], radii=0.08)
        )

    def log_inplane_place_data(self, place_node_positions):
        # log the place nodes with color red
        rr.log(
            "world/inplane_place_nodes",
            rr.Points3D(place_node_positions, colors=[244, 5, 244], radii=0.09)
        )

    def log_bb_data(self, bb_info):
        rr.log(
            "/world/annotations/bb",
            rr.Boxes3D(
                half_sizes=bb_info['bb_half_sizes'],
                centers=bb_info['bb_centroids'],
                labels=bb_info['bb_labels'],
                colors=bb_info['bb_colors']
            ),
            rr.InstancePoses3D(mat3x3=bb_info['bb_mat3x3']),
            timeless=False,
        )

    def log_img_data(self, data):
        # log the camera transform, rgb image, and depth image
        # rr.log("world/agent", rr.Transform3D(transform=camera_from_world))
        # rr.log("world/agent", rr.Pinhole(image_from_camera=intrinsic, resolution=[w, h]))
        rr.log(f"{self.primary_agent_entity}/rgb", rr.Image(data.rgb).compress(jpeg_quality=95))
        rr.log(f"{self.primary_agent_entity}/semantic", rr.Image(data.colormap(data.labels)).compress(jpeg_quality=95))

    def log_hydra_graph(self, is_node=True, node_type='object', nodeid=None, edgeid=None, edge_type='room_to_place', node_pos_source=None, node_pos_target=None):
        if is_node:
            rr.log(
                f"world/hydra_graph/nodes/{node_type}/{nodeid}",
                rr.Points3D(node_pos_source, colors=self._node_color_map[node_type], radii=0.09)
            )
        else: # edge
            rr.log(f"world/hydra_graph/edges/{edge_type}/{edgeid}", rr.Arrows3D(
                origins=node_pos_source,  # Base position of the arrow
                vectors=(node_pos_target-node_pos_source),  # Direction and length of the arrow
                colors=self._edge_color_map[edge_type]
            ))

    def step(self):
        self._t += self._dt
        rr.set_time_seconds(self._timeline, self._t)

    def log_clear(self, namespace):
        rr.log(namespace, rr.Clear(recursive=True))