import rerun as rr
import rerun.blueprint as rrb

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
        rr.log("world/robot_traj", rr.LineStrips3D(agent_positions, colors=[0, 0, 255]))
        rr.log("world/robot", rr.Points3D(agent_positions[-1], colors=[0, 0, 255], radii=0.11))

    def log_traj_data(self, agent_positions):
        rr.log("world/desired_traj", rr.LineStrips3D(agent_positions, colors=[0, 255, 255]))

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

    def log_frontier_data(self, frontier_node_positions):
        # log the frontier nodes with color red
        rr.log(
            "world/frontier_nodes",
            rr.Points3D(frontier_node_positions, colors=[255,0,0], radii=0.08)
        )

    def log_inplane_frontier_data(self, frontier_node_positions):
        # log the frontier nodes with color red
        rr.log(
            "world/inplane_frontier_nodes",
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
            "world/bb",
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

    def step(self):
        self._t += self._dt
        rr.set_time_seconds(self._timeline, self._t)