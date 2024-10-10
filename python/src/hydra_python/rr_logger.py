import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from voxel_mapping.utils.voxel import occupancy_map_to_3d_points

class RRLogger:
    def __init__(self, output_path):
        # Initialize Rerun and specify the .rrd file for logging
        full_output_path = output_path / "test_logger.rrd"

        self._timeline = "vlm_plan_logging"
        rr.init(self._timeline)
        rr.save(full_output_path)

        self.primary_camera_entity = "world/camera"

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
                    origin=self.primary_camera_entity,
                    contents=["$origin/rgb", "/world/annotations/**"],
                ),
                rrb.Spatial2DView(
                        name="Semantic Labels",
                        origin=self.primary_camera_entity,
                        contents=["$origin/semantic", "/world/annotations/**"],
                ),
            ),
            # rrb.Vertical(
            #     rrb.Spatial2DView(
            #         name="Occupied",
            #         origin=self.primary_camera_entity,
            #         contents=["$origin/unoccupied", "/world/annotations/**"],
            #     ),
            #     rrb.Spatial2DView(
            #             name="Explored",
            #             origin=self.primary_camera_entity,
            #             contents=["$origin/unexplored", "/world/annotations/**"],
            #     ),
            #     rrb.Spatial2DView(
            #             name="TSDF",
            #             origin=self.primary_camera_entity,
            #             contents=["$origin/tsdf", "/world/annotations/**"],
            #     ),
            # )
                
        )

        rr.send_blueprint(blueprint)

        self._node_color_map = {
            'object': [225,225,0],
            'frontier': [255,0,0],
            'frontier_selected': [255,255,0],
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
            'frontier-to-object': [255,255,0],
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
        rr.log(f"world/agent_tf", agent_from_world)
        
    def log_camera_tf(self, pos, quat):
        translation = np.asarray([pos[0], pos[1], pos[2]])
        quat_mod = np.asarray([quat[1], quat[2], quat[3], quat[0]])
        camera_from_world = rr.Transform3D(
            translation=translation, rotation=rr.Quaternion(xyzw=quat_mod), from_parent=False
        )
        rr.log(f"{self.primary_camera_entity}", camera_from_world)

    def log_target_poses(self, target_poses):
        rr.log("world/target_poses", rr.Points3D(target_poses, colors=[0,255,0], radii=0.11))
    
    def log_nodes_paths(self, nodes_paths):
        rr.log(f"world/desired_node_path", rr.Points3D(nodes_paths, colors=[255, 192, 203], radii=0.11)) # pink
        rr.log("world/desired_node_path_edges", rr.LineStrips3D(nodes_paths, colors=[255, 192, 203])) # pink

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
        rr.log(f"{self.primary_camera_entity}/rgb", rr.Image(data.rgb).compress(jpeg_quality=95))
        rr.log(f"{self.primary_camera_entity}/semantic", rr.Image(data.colormap(data.labels)).compress(jpeg_quality=95))

    def log_2d_frontier_data(self, unoccupied, unexplored, tsdf):
        rr.log(f"{self.primary_camera_entity}/unoccupied", rr.Image(unoccupied).compress(jpeg_quality=95))
        rr.log(f"{self.primary_camera_entity}/unexplored", rr.Image(unexplored).compress(jpeg_quality=95))
        rr.log(f"{self.primary_camera_entity}/tsdf", rr.Image(tsdf).compress(jpeg_quality=95))

    def log_3d_frontier_data(self, unoccupied_reachable_normal, frontiers_normal, frontiers_unoccupied):
        rr.log(f"world/tsdf_unoccupied", rr.Points3D(unoccupied_reachable_normal, colors=[255, 0, 0], radii=0.06))
        rr.log(f"world/tsdf_frontiers", rr.Points3D(frontiers_normal, colors=[255, 255, 255], radii=0.08))
        rr.log(f"world/tsdf_explored", rr.Points3D(frontiers_unoccupied, colors=[200, 180, 150], radii=0.08))

    def log_voxel_map(
        self,
        space,
        debug: bool = False,
        explored_radius=0.01,
        obstacle_radius=0.05,
    ):
        """Log voxel map and send it to Rerun visualizer
        Args:
            space (SparseVoxelMapNavigationSpace): Voxel map object
        """

        points, _, _, rgb = space.voxel_map.voxel_pcd.get_pointcloud()
        if rgb is None:
            return

        rr.log(
            "world/voxel/point_cloud",
            rr.Points3D(positions=points, radii=np.ones(rgb.shape[0]) * 0.01, colors=np.int64(rgb)),
        )

        grid_origin = space.voxel_map.grid_origin
        obstacles, explored = space.voxel_map.get_2d_map()
        frontier, outside_frontier, traversible = space.get_frontier()

        # self.log_2d_frontier_data(obstacles.detach().cpu().numpy()*255, explored.detach().cpu().numpy()*255, obstacles.detach().cpu().numpy()*255)

        # Get obstacles and explored points
        grid_resolution = space.voxel_map.grid_resolution
        obs_points = np.array(occupancy_map_to_3d_points(obstacles, grid_origin, grid_resolution))
        obs_points_sample_idx = np.random.choice(obs_points.shape[0], size=50, replace=False)
        obs_points_sample = obs_points[obs_points_sample_idx]

        # Get explored points
        explored_points = np.array(occupancy_map_to_3d_points(explored, grid_origin, grid_resolution))
        explored_points_sample_idx = np.random.choice(explored_points.shape[0], size=50, replace=False)
        explored_points_sample = explored_points[explored_points_sample_idx]

        frontier_points = np.array(occupancy_map_to_3d_points(frontier, grid_origin, grid_resolution))
        frontier_points_sample_idx = np.random.choice(frontier_points.shape[0], size=100, replace=False)
        frontier_points_sample = frontier_points[frontier_points_sample_idx]

        # frontier_points_sample = frontier_points.copy()

        outside_frontier_points = np.array(occupancy_map_to_3d_points(outside_frontier, grid_origin, grid_resolution))
        outside_frontier_points_sample_idx = np.random.choice(outside_frontier_points.shape[0], size=20, replace=False)
        outside_frontier_points_sample = outside_frontier_points[outside_frontier_points_sample_idx]
        outside_frontier_points_sample = outside_frontier_points.copy()

        traversible_points = np.array(occupancy_map_to_3d_points(traversible, grid_origin, grid_resolution))
        traversible_points_sample_idx = np.random.choice(traversible_points.shape[0], size=100, replace=False)
        traversible_points_sample = traversible_points[traversible_points_sample_idx]
        # traversible_points_sample = traversible_points.copy()

        # TODO(blake): subsample all of these
        # Log points
        rr.log(
            "world/voxel/obstacles",
            rr.Points3D(
                positions=obs_points,
                radii=np.ones(points.shape[0]) * obstacle_radius,
                colors=[255, 0, 0],
            ),
        )
        rr.log(
            "world/voxel/explored",
            rr.Points3D(
                positions=explored_points,
                radii=np.ones(points.shape[0]) * 0.04,
                colors=[0, 255, 255],
            ),
        )
        rr.log(
            "world/voxel/frontier",
            rr.Points3D(
                positions=frontier_points,
                radii=np.ones(points.shape[0]) * obstacle_radius,
                colors=[255, 255, 0],
            ),
        )
        rr.log(
            "world/voxel/outside_frontier",
            rr.Points3D(
                positions=outside_frontier_points,
                radii=np.ones(points.shape[0]) * 0.04,
                colors=[255, 255, 255],
            ),
        )
        rr.log(
            "world/voxel/traversible",
            rr.Points3D(
                positions=traversible_points,
                radii=np.ones(points.shape[0]) * 0.04,
                colors=[0, 0, 255],
            ),
        )
    
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