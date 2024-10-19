"""Habitat-specific simulator."""
from scipy.spatial.transform import Rotation as R
from typing import Union
import spark_dsg.mp3d
import hydra_python as hydra
import networkx as nx
import numpy as np
import habitat_sim
import pathlib
import magnum
import random
import os
import yaml

from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from hydra_python.frontier_mapping_eqa.utils import *
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel

MISSING_ADE_LABELS = [29, 33]


def _get_index(vertices, pos):
    distances = np.linalg.norm(vertices - np.squeeze(pos), axis=1)
    return np.argmin(distances)


def _compute_path_distance(G, nodes):
    dist = 0.0
    for i in range(len(nodes) - 1):
        v1 = G.nodes[nodes[i]]["pos"]
        v2 = G.nodes[nodes[i + 1]]["pos"]
        dist += np.linalg.norm(v1 - v2)

    return dist

def _build_navgraph(sim, pathfinder, settings, threshold):
    success = sim.recompute_navmesh(pathfinder, settings)
    if not success:
        raise RuntimeError("Failed to make navmesh")

    faces = pathfinder.build_navmesh_vertices()

    G = nx.Graph()

    vertices = []
    for face_vertex in faces:
        curr_pos = np.array(face_vertex).reshape((3, 1))
        matches_prev = False

        for vertex in vertices:
            dist = np.linalg.norm(curr_pos - vertex)
            if dist < threshold:
                matches_prev = True
                break

        if matches_prev:
            continue

        G.add_node(len(vertices), pos=curr_pos)
        vertices.append(curr_pos)

    vertices = np.squeeze(np.array(vertices))
    for i in range(0, len(faces), 3):
        v1 = _get_index(vertices, faces[i + 0])
        v2 = _get_index(vertices, faces[i + 1])
        v3 = _get_index(vertices, faces[i + 2])

        w1 = np.linalg.norm(vertices[v1, :] - vertices[v2, :])
        w2 = np.linalg.norm(vertices[v2, :] - vertices[v3, :])
        w3 = np.linalg.norm(vertices[v3, :] - vertices[v1, :])

        G.add_edge(v1, v2, weight=w1)
        G.add_edge(v2, v3, weight=w2)
        G.add_edge(v3, v1, weight=w3)

    largest_cc = max(nx.connected_components(G), key=len)
    # return G.subgraph(largest_cc)
    return G


def _make_sensor(sensor_type, width=640, height=360, hfov=90.0, camera_height=0.0):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = str(sensor_type)
    spec.sensor_type = sensor_type
    spec.resolution = [height, width]
    spec.position = [0.0, camera_height, 0.0]
    spec.orientation = [0.0, 0.0, 0.0]
    spec.hfov = hfov
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    return spec


def _make_habitat_config(scene, scene_type='mp3d', camera_height=0.0, width=640, height=360, agent_z_offset=0.0, agent_radius=0.1, hfov=90.0):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    path = scene.parent.parent
    if scene_type=='mp3d':
        json_path = path / "mp3d.scene_dataset_config.json"
    elif scene_type=='hm3d':
        json_path = path / "hm3d_annotated_train_basis.scene_dataset_config.json"
    else:
        raise NotImplementedError('scene type not implemented.')

    sim_cfg.scene_dataset_config_file = str(json_path)
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = str(scene)
    sim_cfg.enable_physics = True
    sim_cfg.allow_sliding = False

    sensor_specs = [
        _make_sensor(x, camera_height=camera_height, width=width, height=height, hfov=hfov)
        for x in [
            habitat_sim.SensorType.COLOR,
            habitat_sim.SensorType.DEPTH,
            habitat_sim.SensorType.SEMANTIC,
        ]
    ]

    camera_spec = sensor_specs[0]
    height, width = camera_spec.resolution
    focal_length = width / (2.0 * np.tan(float(camera_spec.hfov) * np.pi / 360.0))
    camera_info = {
        "fx": float(focal_length),
        "fy": float(focal_length),
        "cx": float(width / 2.0),
        "cy": float(height / 2.0),
        "width": int(width),
        "height": int(height),
    }

    agent_cfg = habitat_sim.agent.AgentConfiguration(
        height=agent_z_offset,
        radius=agent_radius,
        sensor_specifications=sensor_specs,
        action_space={},
        body_type="cylinder",
    )

    return habitat_sim.Configuration(sim_cfg, [agent_cfg]), camera_info


def _set_logging():
    log_settings = [
        "Default=Quiet",
        "Metadata=Quiet",
        "Assets=Quiet",
        "Physics=Quiet",
        "Nav=Quiet",
        "Scene=Quiet",
        "Sensor=Quiet",
        "Gfx=Quiet",
    ]
    os.environ["HABITAT_SIM_LOG"] = ":".join(log_settings)
    try:
        habitat_sim._ext.habitat_sim_bindings.core.LoggingContext.reinitialize_from_env()
    except AttributeError:
        print("Failed to setup logging")


def _transform_from_habitat(h_q_c, p_h):
    w_R_h = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    p_w = np.squeeze(w_R_h @ p_h.reshape((3, 1)))

    # R.from_quat expects xyzw order by default
    h_R_c = R.from_quat(h_q_c).as_matrix()
    w_R_c = w_R_h @ h_R_c @ w_R_h.T
    w_q_c = R.from_matrix(w_R_c).as_quat() # returns xyzw format
    return w_q_c, p_w


def _transform_to_habitat(w_q_c, p_w):
    h_R_w = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    p_h = np.squeeze(h_R_w @ p_w.reshape((3, 1)))

    w_R_c = R.from_quat(w_q_c).as_matrix()
    h_R_c = h_R_w @ w_R_c @ h_R_w.T
    h_q_c = R.from_matrix(h_R_c).as_quat()
    return h_q_c, p_h


def _transform_to_body(w_q_c, p_c):
    b_R_c = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    w_R_c = R.from_quat(w_q_c).as_matrix()
    w_R_b = w_R_c @ b_R_c.T
    w_q_b = R.from_matrix(w_R_b).as_quat()
    # note that b_T_c doesn't have translation...
    return w_q_b, p_c

def _transform_from_body(w_q_c, p_c):
    b_R_c = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    w_R_c = R.from_quat(w_q_c).as_matrix()
    w_R_b = w_R_c @ b_R_c.T
    w_q_b = R.from_matrix(w_R_b).as_quat()
    # note that b_T_c doesn't have translation...
    return w_q_b, p_c


def _camera_point_from_habitat(p_ah, z_offset=1.5):
    p_bh = np.array(p_ah)
    # z offset is relative to ENU
    p_bh[1] += z_offset
    bw_R_bh = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    p_bw = np.squeeze(bw_R_bh @ p_bh.reshape((3, 1)))

    return p_bw

def _habitat_to_world_eqa(p_h):
    bw_R_bh = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    # bw_R_bh = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    # p_w = np.squeeze(bw_R_bh @ p_h.reshape((3, 1)))
    quat_xyzw, pos = _transform_from_habitat(R.from_matrix(bw_R_bh).as_quat(), np.array(p_h))
    quat_xyzw_normal, pos_normal = _transform_from_body(quat_xyzw, pos)
    return pos_normal

def _camera_point_to_habitat(p_camera, z_offset=1.5):
    p_habitat = p_camera.copy()
    p_habitat[2] -= z_offset
    bw_R_c = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    p_habitat = np.squeeze(bw_R_c @ p_habitat.reshape((3, 1)))
    return p_habitat

def _angle_to_rotation_habitat(angle, camera_tilt_deg):
    camera_tilt = camera_tilt_deg * np.pi / 180
    rotation_xyzw = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
    return rotation_xyzw

def _format_list(name, values, collapse=True, **kwargs):
    indent = kwargs.get("indent", 0)
    prefix = " " * indent + name if indent > 0 else name
    if collapse:
        indent += 2

    args = {
        k: v for k, v in kwargs.items() if k != "indent" and k != "default_flow_style"
    }
    args["indent"] = indent
    value_str = yaml.dump(values, default_flow_style=collapse, **args)
    return f"{prefix}: {value_str}"

class HabitatInterface:
    """Class handling interfacing with habitat."""

    def __init__(
            self, 
            scene: Union[str, pathlib.Path], 
            scene_type: str='mp3d',
            inflation_radius=0.25,
            camera_height=0.0, 
            width=640, 
            height=360,
            agent_z_offset=0.0,
            hfov=90.0,
            z_offset=0.5,
            camera_tilt=0,
            get_clip_embeddings=False,
            get_siglip_embeddings=False,
            img_subsample_freq=1):
        
        """Initialize the simulator."""
        scene = pathlib.Path(scene).expanduser().resolve()
        self._scene_type = scene_type
        self.inflation_radius = inflation_radius
        self.z_offset = z_offset
        self._camera_tilt = camera_tilt
        self.question = ' '
        self._get_clip_embeddings = get_clip_embeddings
        self._get_siglip_embeddings = get_siglip_embeddings
        self._img_subsample_freq = img_subsample_freq

        # TODO(nathan) expose some of this via the data interface
        _set_logging()
        config, camera_info = _make_habitat_config(
            scene, 
            scene_type=scene_type, 
            camera_height=camera_height,
            width=width, 
            height=height,
            agent_z_offset=agent_z_offset, 
            agent_radius=0.1, 
            hfov=hfov)
        self._house_path = scene.parent / f"{scene.stem}.house"
        self._camera_info = camera_info
        
        self._sim = habitat_sim.Simulator(config)
        
        if scene_type=='mp3d':
            self._make_instance_labelmap_mp3d()
        if scene_type=='hm3d':
            self._make_instance_labelmap_hm3d()
            # self._write_config_yaml()

        self._obs = None
        self._labels = None

        if get_clip_embeddings:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.question_embed = self.processor(text=[self.question], return_tensors="pt", padding=True)

        if get_siglip_embeddings:
            self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
            self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            self.question_embed = self.processor(text=[self.question], padding="max_length", return_tensors="pt")

        self._make_navgraph(inflation_radius=inflation_radius)

    def update_question(self, question):
        self.question = question
        if self._get_clip_embeddings:
            self.question_embed = self.processor(text=[question], return_tensors="pt", padding=True)
        if self._get_siglip_embeddings:
            self.question_embed = self.processor(text=[question], return_tensors="pt", padding="max_length")
    
    def calc_logit_for_img(self, img):
        imgs_embed = self.processor(images=img, return_tensors="pt", padding=True)
        outputs = self.model(**self.question_embed, **imgs_embed)
        logits_per_text = outputs.logits_per_image # this is the image-text similarity score
        return logits_per_text.squeeze().detach().numpy()

    def calc_similarity_score(self, images):
        padding = True if self._get_clip_embeddings else "max_length" # HuggingFace says SigLIP was trained on "max_length"
        imgs_embed = self.processor(images=images[::self._img_subsample_freq], return_tensors="pt", padding=padding)
        outputs = self.model(**self.question_embed, **imgs_embed)
        logits_per_text = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_text.softmax(dim=0).squeeze() # we can take the softmax to get the label probabilities
        return probs.detach().numpy(), logits_per_text.squeeze().detach().numpy()

    def _make_instance_labelmap_mp3d(self):
        object_to_cat_map = {
            c.id: c.category.index() for c in self._sim.semantic_scene.objects
        }

        mpcat_to_ade = np.arange(41)
        for label in sorted(MISSING_ADE_LABELS):
            mpcat_to_ade[label:] -= 1
            mpcat_to_ade[label] = 0

        ade_to_mpcat = {}
        for mpcat_idx, ade_idx in enumerate(mpcat_to_ade):
            if ade_idx not in ade_to_mpcat:
                ade_to_mpcat[ade_idx] = mpcat_idx

        category_map = np.array(
            [mpcat_to_ade[idx] for _, idx in object_to_cat_map.items()]
        )
        self._labelmap = hydra.LabelConverter(category_map)

        name_mapping = {}
        for c in self._sim.semantic_scene.categories:
            name_mapping[c.index()] = c.name()

        keys = sorted([x for x in ade_to_mpcat])
        names = [name_mapping[ade_to_mpcat[idx]] for idx in keys]
        self._colormap = hydra.SegmentationColormap.from_names(names=names)
    
    def _make_instance_labelmap_hm3d(self):
        object_to_cat_map = {
            c.id: c.category.index() for c in self._sim.semantic_scene.objects
        }

        category_map = np.array(list(object_to_cat_map.values()))
        self._labelmap = hydra.LabelConverter(category_map)

        name_mapping = {}
        for c in self._sim.semantic_scene.categories:
            name_mapping[c.index()] = c.name()

        hm3d_cat_idxs = sorted(list(name_mapping.keys()))
        names = [name_mapping[idx] for idx in hm3d_cat_idxs]
        self._colormap = hydra.SegmentationColormap.from_names(names=names)

    def _write_config_yaml(self):
        output_path = pathlib.Path("/home/saumyas/catkin_ws_semnav/src/hydra/config/label_spaces/hm3d_label_space.yaml")
        invalid_labels = []
        surface_labels = []
        dynamic_labels = []
        object_labels = []
        output_names = []

        for id, name in enumerate(self._colormap.names):
            output_names.append({"label": id, "name": name})
            if 'unknown' in name.lower():
                invalid_labels.append(id)
            elif 'floor' in name.lower():
                surface_labels.append(id)
            else:
                object_labels.append(id)

        with output_path.open("w") as fout:
            fout.write("---\n")

            fout.write(yaml.dump({"total_semantic_labels": len(self._colormap.names)}))
            fout.write(_format_list("dynamic_labels", dynamic_labels))
            fout.write(_format_list("invalid_labels", invalid_labels))
            fout.write("object_labels:\n")
            for name in object_labels:
                fout.write("  - " + yaml.dump(name, default_flow_style=True))
            fout.write(_format_list("surface_places_labels", surface_labels))

            fout.write("label_names:\n")
            for name in output_names:
                fout.write("  - " + yaml.dump(name, default_flow_style=True))

    def _make_navgraph(self, inflation_radius=0.1, threshold=1.0e-3):
        self.pathfinder = habitat_sim.nav.PathFinder()
        settings = habitat_sim.NavMeshSettings()
        settings.agent_radius = inflation_radius
        self.G = _build_navgraph(self._sim, self.pathfinder, settings, threshold)
    
    def get_full_trajectory(
        self,
        seed=None,
        add_reverse=False,
        max_room_distance=5.0,
        **kwargs,
    ):
        """Get a trajectory that explores the entire scene."""

        components = list(nx.connected_components(self.G))
        if len(components) > 1:
            print("Warning: {len(components)} components found in navgraph!")
            components = sorted(components, lambda x: len(x), reverse=True)

        mp3d_info = spark_dsg.mp3d.load_mp3d_info(self._house_path)
        rooms = spark_dsg.mp3d.get_rooms_from_mp3d_info(mp3d_info, angle_deg=-90)

        node_sequence = []
        for room in rooms:
            best_node = None
            best_distance = None
            for x in components[0]:
                pos = _camera_point_from_habitat(self.G.nodes[x]["pos"], z_offset=self.z_offset)
                dist = np.linalg.norm(pos - room.centroid)
                if not best_distance or dist < best_distance:
                    best_node = x
                    best_distance = dist

            if best_node is None:
                continue

            if best_distance > max_room_distance:
                pos = _camera_point_from_habitat(
                    self.G.nodes[best_node]["pos"], z_offset=self.z_offset
                )
                print(f"No node for {room.get_id()} @ {np.squeeze(room.centroid)}")
                print(f"Closest: {best_node} @ {np.squeeze(pos)}")
                continue

            node_sequence.append(best_node)

        if len(node_sequence) <= 1:
            return None

        if seed is not None:
            random.seed(seed)

        random.shuffle(node_sequence)
        if add_reverse:
            node_sequence = node_sequence + node_sequence[::-1]

        b_R_c = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        first_pos_habitat = self.G.nodes[node_sequence[0]]["pos"]
        first_pos_cam = _camera_point_from_habitat(first_pos_habitat, z_offset=self.z_offset)

        traj = hydra.Trajectory.rotate(first_pos_cam, body_R_camera=b_R_c, **kwargs)
        for i in range(len(node_sequence) - 1):
            start = node_sequence[i]
            end = node_sequence[i + 1]
            nodes = nx.shortest_path(self.G, source=start, target=end, weight="weight")
            if len(nodes) <= 1:
                continue

            pos_habitat = [self.G.nodes[x]["pos"] for x in nodes]
            pos_cam = [
                _camera_point_from_habitat(p, z_offset=self.z_offset) for p in pos_habitat
            ]
            new_traj = hydra.Trajectory.from_positions(
                np.array(pos_cam), body_R_camera=b_R_c, **kwargs
            )

            traj += new_traj
            traj += hydra.Trajectory.rotate(
                np.array(pos_cam[-1]), body_R_camera=b_R_c, **kwargs
            )

        return traj
    

    def get_random_trajectory(
        self,
        target_length_m=100.0,
        seed=None,
    ):
        """Get a trajectory as sequence of segments between random areas in a scene."""

        if seed is not None:
            random.seed(seed)

        node_sequence = [x for x in self.G]
        random.shuffle(node_sequence)

        path = []
        total_length = 0.0
        for i in range(len(node_sequence) - 1):
            start = node_sequence[i]
            end = node_sequence[i + 1]
            nodes = nx.shortest_path(self.G, source=start, target=end, weight="weight")
            total_length += _compute_path_distance(self.G, nodes)
            path += nodes[:-1]

            if total_length > target_length_m:
                break

        positions_habitat = [self.G.nodes[x]["pos"] for x in path]
        positions_camera = [
            _camera_point_from_habitat(p, z_offset=self.z_offset) for p in positions_habitat
        ]
        b_R_c = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        return hydra.Trajectory.from_positions(
            np.array(positions_camera), body_R_camera=b_R_c
        )
    
    def get_trajectory_to_pose(self, start, end):
        """Get a trajectory from start to end in navgraph G."""

        start_habitat = _camera_point_to_habitat(start, z_offset=self.z_offset)
        end_habitat = _camera_point_to_habitat(end, z_offset=self.z_offset)

        # Find closest node on graph
        node_sequence = [x for x in self.G]
        pos_nodes = np.array([self.G.nodes[x]["pos"] for x in self.G]).squeeze()

        start_idx = np.argmin(np.linalg.norm(pos_nodes - start_habitat, axis=-1))
        end_idx = np.argmin(np.linalg.norm(pos_nodes - end_habitat, axis=-1))

        nodes = nx.shortest_path(self.G, source=node_sequence[start_idx], target=node_sequence[end_idx], weight="weight")

        positions_habitat = [self.G.nodes[x]["pos"] for x in nodes]
        
        if len(positions_habitat) < 2:
            return None

        positions_camera = [
            _camera_point_from_habitat(p, z_offset=self.z_offset) for p in positions_habitat
        ]
        b_R_c = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        poses = hydra.Trajectory.from_positions(
            np.array(positions_camera), body_R_camera=b_R_c
        )
        return poses
    
    def get_trajectory_to_pose_habitat_eqa(self, target_pos):
        """Get a trajectory from target_pos in navgraph G.
        target_pos is taken from the hydra scenegraph"""

        agent = self._sim.get_agent(0)  # Assuming agent ID 0
        current_pos = agent.get_state().position
        current_quat_wxyz = agent.get_state().rotation

        end_habitat = _camera_point_to_habitat(target_pos, z_offset=self.z_offset)

        # Find closest node on graph
        node_sequence = [x for x in self.G]
        pos_nodes = np.array([self.G.nodes[x]["pos"] for x in self.G]).squeeze()

        start_idx = np.argmin(np.linalg.norm(pos_nodes - current_pos, axis=-1))
        end_idx = np.argmin(np.linalg.norm(pos_nodes - end_habitat, axis=-1))

        nodes = nx.shortest_path(self.G, source=node_sequence[start_idx], target=node_sequence[end_idx], weight="weight")

        positions_habitat = [self.G.nodes[x]["pos"] for x in nodes]
        
        if len(positions_habitat) < 2:
            return None

        positions_camera = [
            _camera_point_from_habitat(p, z_offset=self.z_offset) for p in positions_habitat
        ]

        quat_xyzw, pos = _transform_from_habitat(quat_to_coeffs(current_quat_wxyz), np.array(current_pos))
        quat_xyzw_normal, pos_normal = _transform_from_body(quat_xyzw, pos)

        b_R_c = R.from_quat(quat_xyzw_normal).as_matrix()
        poses = hydra.Trajectory.from_positions(
            np.array(positions_camera), body_R_camera=b_R_c
        )
        return poses

    def get_trajectory_to_pose_world_eqa(self, target_pos, navmesh_sg):
        """Get a trajectory from target_pos in navgraph G.
        target_pos is taken from the hydra scenegraph"""

        agent = self._sim.get_agent(0)  # Assuming agent ID 0
        current_pos = agent.get_state().position
        current_quat_wxyz = agent.get_state().rotation

        quat_xyzw, pos = _transform_from_habitat(quat_to_coeffs(current_quat_wxyz), np.array(current_pos))
        current_quat_xyzw, current_pos = _transform_from_body(quat_xyzw, pos)
        current_quat_wxyz = np.roll(current_quat_xyzw, 1)

        # Find closest node on graph
        node_sequence = [x for x in navmesh_sg]
        pos_nodes = np.array([navmesh_sg.nodes[x]["position"] for x in navmesh_sg]).squeeze()

        start_idx = np.argmin(np.linalg.norm(pos_nodes - current_pos, axis=-1))
        end_idx = np.argmin(np.linalg.norm(pos_nodes - target_pos, axis=-1))

        nodes = nx.shortest_path(navmesh_sg, source=node_sequence[start_idx], target=node_sequence[end_idx], weight="weight")

        positions_camera = [navmesh_sg.nodes[x]["position"] for x in nodes]
        positions_camera_proj = np.array(positions_camera)
        positions_camera_proj[:,2] = current_pos[2] # project to agent plane
        
        if positions_camera_proj.shape[0] < 2:
            return None

        b_R_c = R.from_quat(current_quat_xyzw).as_matrix()
        # poses = hydra.Trajectory.from_positions(
        #     np.array(positions_camera), body_R_camera=b_R_c
        # )
        poses = hydra.Trajectory.from_positions_eqa(
            np.concatenate([current_pos.reshape(1,3), positions_camera_proj], axis=0), init_quat_wxyz=current_quat_wxyz
        )
        return poses, positions_camera

    def get_trajectory_from_path_habitat_frame(self, desired_path):
        # desired path is in world frame of eqa
        desired_path_habitat = pos_normal_to_habitat(desired_path)

        agent = self._sim.get_agent(0)  # Assuming agent ID 0
        current_pos = agent.get_state().position
        current_quat_xyzw = quat_to_coeffs(agent.get_state().rotation)
        current_quat_wxyz = np.roll(current_quat_xyzw, 1)

        desired_path_habitat[:,1] = current_pos[1] # project to agent plane, check
        poses = hydra.Trajectory.from_positions_habitat(
            np.concatenate([current_pos.reshape(1,3), desired_path_habitat], axis=0), init_quat_wxyz=current_quat_wxyz
        )
        return poses

    def get_trajectory_from_path_habitat_frame2(self, path_normal, current_heading, camera_tilt_deg):
        agent = self._sim.get_agent(0)  # Assuming agent ID 0
        current_pos = agent.get_state().position
        current_quat_xyzw = quat_to_coeffs(agent.get_state().rotation)
        current_quat_wxyz = np.roll(current_quat_xyzw, 1)

        # desired path is in world frame of eqa
        desired_path_habitat = pos_normal_to_habitat(path_normal)
        diff = desired_path_habitat[-1] - desired_path_habitat[0]
        desired_heading = np.arctan2(-diff[0],-diff[2])
        desired_quat_habitat_xyzw = _angle_to_rotation_habitat(desired_heading, camera_tilt_deg)
        desired_quat_habitat_wxyz = np.roll(desired_quat_habitat_xyzw, 1)

        yaw_diff = abs((current_heading - desired_heading + np.pi) % (2 * np.pi) - np.pi)
        desired_quat_habitat_wxyz, yaw_diff = [], []

        yaw_prev = current_heading
        pos_prev = current_pos.copy()
        for i in range(len(desired_path_habitat)):
            diff = desired_path_habitat[i] - pos_prev
            desired_heading = np.arctan2(-diff[0],-diff[2])
            # heading_sample_range = [desired_heading-30*np.pi/180, desired_heading+30*np.pi/180]
            # desired_heading = np.random.uniform(heading_sample_range[0], heading_sample_range[1])
            des_quat_xyzw = _angle_to_rotation_habitat(desired_heading, camera_tilt_deg)
            desired_quat_habitat_wxyz.append(np.roll(des_quat_xyzw, 1))
            yaw_diff.append(abs((desired_heading - yaw_prev + np.pi) % (2 * np.pi) - np.pi))
            yaw_prev = desired_heading
            pos_prev = desired_path_habitat[i].copy()

        desired_path_habitat[:,1] = current_pos[1] # project to agent plane, check
        poses = hydra.Trajectory.from_poses_habitat_yaw(
            np.concatenate([current_pos.reshape(1,3), desired_path_habitat], axis=0), 
            init_quat_wxyz=current_quat_wxyz,
            desired_quat_wxyz=desired_quat_habitat_wxyz,
            yaw_diff=yaw_diff
        )
        return poses
    
    def get_trajectory_from_path_angles_habitat_frame(self, path_normal, angles, current_heading, camera_tilt_deg):
        # desired path is in world frame of eqa
        desired_path_habitat = pos_normal_to_habitat(path_normal)
        diff = desired_path_habitat[-1] - desired_path_habitat[0]
        desired_heading = np.arctan2(-diff[0],-diff[2])
        desired_quat_habitat_xyzw = _angle_to_rotation_habitat(desired_heading, camera_tilt_deg)
        desired_quat_habitat_wxyz = np.roll(desired_quat_habitat_xyzw, 1)

        yaw_diff = abs((current_heading - desired_heading + np.pi) % (2 * np.pi) - np.pi)
        
        # heading_sample_range = [desired_heading-30*np.pi/180, desired_heading+30*np.pi/180]
        # desired_quat_habitat_wxyz, yaw_diff = [], []
        # yaw_prev = current_heading
        # for i in len(desired_path_habitat):
        #     heading_angle = np.random.uniform(heading_sample_range[0], heading_sample_range[1])
        #     des_quat_xyzw = _angle_to_rotation_habitat(heading_angle, camera_tilt_deg)
        #     desired_quat_habitat_wxyz.append(np.roll(des_quat_xyzw, 1))
        #     yaw_diff.append(abs((heading_angle - yaw_prev + np.pi) % (2 * np.pi) - np.pi))
        #     yaw_prev = heading_angle

        agent = self._sim.get_agent(0)  # Assuming agent ID 0
        current_pos = agent.get_state().position
        current_quat_xyzw = quat_to_coeffs(agent.get_state().rotation)
        current_quat_wxyz = np.roll(current_quat_xyzw, 1)

        desired_path_habitat[:,1] = current_pos[1] # project to agent plane, check
        poses = hydra.Trajectory.from_poses_habitat(
            np.concatenate([current_pos.reshape(1,3), desired_path_habitat], axis=0), 
            init_quat_wxyz=current_quat_wxyz,
            desired_quat_wxyz=desired_quat_habitat_wxyz,
            yaw_diff=yaw_diff
        )
        return poses
    
    def get_init_poses_eqa(self, pos_hab, angle, camera_tilt_deg):
        # pose is in habitat frame, return (pose, quat_wxyz)
        quat_habitat_xyzw = _angle_to_rotation_habitat(angle, camera_tilt_deg)
        quat_habitat_wxyz = np.roll(quat_habitat_xyzw, 1)

        poses = []
        dt = 0.2
        for i in range(10):
            poses.append((int(i*dt*1e9), pos_hab, quat_habitat_wxyz))
        return poses

    def get_init_poses_hydra(self, pos_hab, angle, camera_tilt_deg):
        # pose is in habitat frame, return (pose, quat_wxyz)
        quat_hab_xyzw = _angle_to_rotation_habitat(angle, camera_tilt_deg)

        # habitat_euler = R.from_quat(quat_hab_xyzw).as_euler('xyz', degrees=True)

        quat_xyzw, pos = _transform_from_habitat(np.array(quat_hab_xyzw), np.array(pos_hab))
        # quat_xyzw_re, pos_re = _transform_to_habitat(quat_xyzw, pos) # testing

        quat_xyzw_normal, pos_normal = _transform_from_body(quat_xyzw, pos)
        # quat_xyzw_normal_re, pos_normal_re = _transform_to_body(quat_xyzw_normal, pos_normal) #testing
        # hydra_euler = R.from_quat(quat_xyzw_normal).as_euler('xyz', degrees=False)

        quat_wxyz_normal = np.roll(quat_xyzw_normal, 1)

        poses = []
        dt = 0.2
        for i in range(10):
            poses.append((int(i*dt*1e9), pos_normal, quat_wxyz_normal))
        return poses
    
    def get_rotate_in_place_trajectory(
        self, 
        seed=None,
    ):
        
        if seed is not None:
            np.random.seed(seed)
            
        start = np.random.choice([x for x in self.G])
        start_pos_habitat = self.G.nodes[start]["pos"]

        position_camera = _camera_point_from_habitat(start_pos_habitat, z_offset=self.z_offset)
        b_R_c = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        return hydra.Trajectory.rotate(
            np.array(position_camera), body_R_camera=b_R_c
        )

    def set_pose(self, timestamp, world_T_camera, is_eqa=False):
        """Set pose of the agent directly."""
        if is_eqa:
            # pose_normal = pose_tsdf_to_normal(world_T_camera)
            # pose_habitat = pose_normal_to_habitat(pose_normal)
            p_h = world_T_camera[:3, 3]
            h_q_c = R.from_matrix(world_T_camera[:3, :3]).as_quat()
        else:
            w_q_c = R.from_matrix(world_T_camera[:3, :3]).as_quat()
            w_q_b, p_b = _transform_to_body(w_q_c, world_T_camera[:3, 3])
            h_q_c, p_h = _transform_to_habitat(w_q_b, p_b)

        new_state = habitat_sim.AgentState()
        new_state.position = magnum.Vector3(p_h[0], p_h[1], p_h[2])
        new_state.rotation = h_q_c
        self._sim.agents[0].set_state(new_state)
        self._obs = self._sim.get_sensor_observations()
        self._labels = None

    def get_heading_angle(self):
        agent = self._sim.get_agent(0)
        current_quat = agent.get_state().rotation
        
        quat_camera_tilt = quat_from_angle_axis(self._camera_tilt, np.array([1, 0, 0]))
        quat_heading_xyzw = quat_to_coeffs(current_quat*quat_camera_tilt.inverse())
        heading_angle = 2 * np.arctan2(quat_heading_xyzw[1], quat_heading_xyzw[3])
        # heading_angle = R.from_quat(quat_to_coeffs(agent.get_state().rotation)).as_euler('xyz', degrees=False)[1]
        return heading_angle
    
    def get_state(self, is_eqa=False):
        agent = self._sim.get_agent(0)  # Assuming agent ID 0
        current_pos = agent.get_state().position
        current_quat_wxyz = agent.get_state().rotation

        if is_eqa:
            pose = np.eye(4)
            pose[:3, :3] = R.from_quat(quat_to_coeffs(current_quat_wxyz)).as_matrix()
            pose[:3, 3] = current_pos
            pose_normal = pose_habitat_to_normal(pose)
            pose_tsdf = pose_normal_to_tsdf(pose_normal)

            current_pos = pose_normal[:3, 3]
            current_quat_xyzw = R.from_matrix(pose_normal[:3, :3]).as_quat()
            current_quat_wxyz = np.roll(current_quat_xyzw, 1)
        else:
            quat_xyzw, pos = _transform_from_habitat(quat_to_coeffs(current_quat_wxyz), np.array(current_pos))
            current_quat_xyzw, current_pos = _transform_from_body(quat_xyzw, pos)
            current_quat_wxyz = np.roll(current_quat_xyzw, 1)
        return current_pos, current_quat_wxyz

    def get_camera_pos(self, is_eqa=False):
        if is_eqa:
            pose_cam = get_cam_pose_tsdf(self.get_depth_sensor_state())
            current_pos = pose_cam[:3, 3]
            q_xyzw = R.from_matrix(pose_cam[:3, :3]).as_quat()
            current_quat_wxyz = np.roll(q_xyzw, 1)
        else:
            pass
        return current_pos, current_quat_wxyz
    
    def get_depth_sensor_state(self):
        agent = self._sim.get_agent(0)
        return agent.get_state().sensor_states[str(habitat_sim.SensorType.DEPTH)]
    
    @property
    def colormap(self):
        """Get colormap between labels and semantic colors."""
        return self._colormap

    @property
    def camera_info(self):
        """Get camera properties."""
        return self._camera_info

    @property
    def depth(self):
        """Get current depth observation if available."""
        if self._obs is None:
            return None

        return self._obs[str(habitat_sim.SensorType.DEPTH)]

    @property
    def labels(self):
        """Get current semantics observation if available."""
        if self._obs is None:
            return None

        if self._labels is None:
            instance_labels = self._obs[str(habitat_sim.SensorType.SEMANTIC)]
            self._labels = self._labelmap(instance_labels).astype(np.int32)

        return self._labels

    @property
    def rgb(self):
        """Get current RGB observation if available."""
        if self._obs is None:
            return None

        return self._obs[str(habitat_sim.SensorType.COLOR)][:, :, :3]


hydra.DataInterface.register(HabitatInterface)
