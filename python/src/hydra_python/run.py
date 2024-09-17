"""Module containing code for running Hydra."""
from scipy.spatial.transform import Rotation as R
import numpy as np
import click


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

    imgs_colormap, imgs_rgb, imgs_labels = [], [], []

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

        labeled_frames.append(color_img)

    imageio.mimsave(output_path / f'images_hm3d_semantic_{suffix}.gif', imgs_colormap)
    imageio.mimsave(output_path / f'images_hm3d_rgb_{suffix}.gif', imgs_rgb)
    imageio.mimsave(output_path / f'images_hm3d_labeled_frame_{suffix}s.gif', labeled_frames)