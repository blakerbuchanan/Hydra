# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This file contains versions of the helpers in point_cloud.py that use pytorch directly (rather than numpy),
to allow operations to be done on the GPU for speed.
"""

from typing import List, Optional, Union

import cv2, gc
import numpy as np
import torch
from torch import Tensor

from voxel_mapping.utils.image import Camera

USE_TORCH_GEOMETRIC = False
# if USE_TORCH_GEOMETRIC:
#     from torch_geometric.nn.pool.voxel_grid import voxel_grid
# else:
#     from stretch.utils.torch_geometric.torch_geometric_helpers import voxel_grid


def depth_to_xyz(depth: torch.Tensor, camera: Camera):
    """get depth from numpy using simple pinhole camera model"""
    # TODO: convert to torch:
    # xs, ys = torch.meshgrid(
    #     torch.arange(0, width), torch.arange(0, height), indexing="xy", device=depth.device
    # )
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(1, 2, 0)
    z = depth

    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    x = torch.tensor(indices[:, :, 1] - camera.px).to(z.device) * (z / camera.fx)
    y = torch.tensor(indices[:, :, 0] - camera.py).to(z.device) * (z / camera.fy)

    # Should now be batch x height x width x 3, after this:
    xyz = torch.stack([x, y, z], dim=-1)
    return xyz


def unproject_masked_depth_to_xyz_coordinates(
    depth: torch.Tensor,
    pose: torch.Tensor,
    inv_intrinsics: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns the XYZ coordinates for a batch posed RGBD image.

    Args:
        depth: The depth tensor, with shape (B, 1, H, W)
        mask: The mask tensor, with the same shape as the depth tensor,
            where True means that the point should be masked (not included)
        inv_intrinsics: The inverse intrinsics, with shape (B, 3, 3)
        pose: The poses, with shape (B, 4, 4)

    Returns:
        XYZ coordinates, with shape (N, 3) where N is the number of points in
        the depth image which are unmasked
    """
    gc.collect()
    torch.cuda.empty_cache()
    batch_size, _, height, width = depth.shape
    if mask is None:
        mask = torch.full_like(depth, fill_value=False, dtype=torch.bool)
    flipped_mask = ~mask

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=depth.device),
        torch.arange(0, height, device=depth.device),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1)[None, :, :].repeat_interleave(batch_size, dim=0)
    xy = xy[flipped_mask.squeeze(1)]
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Associates poses and intrinsics with XYZ coordinates.
    inv_intrinsics = inv_intrinsics[:, None, None, :, :].expand(batch_size, height, width, 3, 3)[
        flipped_mask.squeeze(1)
    ]
    pose = pose[:, None, None, :, :].expand(batch_size, height, width, 4, 4)[
        flipped_mask.squeeze(1)
    ]
    depth = depth[flipped_mask]

    # Applies intrinsics and extrinsics.
    xyz = xyz.to(inv_intrinsics).unsqueeze(1) @ inv_intrinsics.permute([0, 2, 1])
    xyz = xyz * depth[:, None, None]
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[..., None, :3, 3]
    xyz = xyz.squeeze(1)

    return xyz


def add_additive_noise_to_xyz(
    xyz_img: torch.Tensor,
    gp_rescale_factor_range: Optional[List[int]] = [12, 20],
    gaussian_scale_range: Optional[List[float]] = [0.0, 0.003],
    valid_mask: Optional[torch.Tensor] = None,
    inplace: Optional[bool] = False,
):
    """
    Add (approximate) Gaussian Process noise to ordered point cloud
    @param xyz_img: a [H x W x 3] ordered point cloud
    """
    if not inplace:
        xyz_img = xyz_img.clone()

    H, W, C = xyz_img.shape

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    gp_rescale_factor = np.random.randint(gp_rescale_factor_range[0], gp_rescale_factor_range[1])
    gp_scale = np.random.uniform(gaussian_scale_range[0], gaussian_scale_range[1])

    small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
    additive_noise = np.random.normal(loc=0.0, scale=gp_scale, size=(small_H, small_W, C))
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)  # type: ignore
    additive_noise = torch.tensor(additive_noise).to(xyz_img.device)  # type: ignore
    if valid_mask is not None:
        xyz_img[valid_mask, :] += additive_noise[valid_mask, :]
    else:
        xyz_img += additive_noise

    return xyz_img


def dropout_random_ellipses(
    depth_img: torch.Tensor,
    dropout_mean: float,
    gamma_shape: Optional[float] = 10000,
    gamma_scale: Optional[float] = 0.0001,
    inplace: Optional[bool] = False,
):
    """Randomly drop a few ellipses in the image for robustness.
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
    @param depth_img: a [H x W] set of depth z values
    """
    if not inplace:
        depth_img = depth_img.clone()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(dropout_mean)

    # Sample ellipse centers
    nonzero_pixel_indices = torch.stack(
        torch.where(depth_img > 0)
    ).T  # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(
        nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout
    )
    dropout_centers = nonzero_pixel_indices[
        dropout_centers_indices, :
    ]  # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(gamma_shape, gamma_scale, size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(gamma_shape, gamma_scale, size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :].cpu().numpy()
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # dropout the ellipse
        # mask is always 2d even if input is not
        mask = np.zeros(depth_img.shape[:2])
        mask = cv2.ellipse(
            mask,
            tuple(center[::-1]),
            (x_radius, y_radius),
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1,
        )  # type: ignore
        depth_img[mask == 1] = 0

    return depth_img


# def get_one_point_per_voxel_from_pointcloud(
#     unbatched_xyz: torch.Tensor,
#     unbatched_batch_ids: torch.Tensor,
#     voxel_size: Union[float, List[float], torch.Tensor],
#     use_random_centers: Optional[bool] = True,
# ) -> torch.Tensor:
#     """
#     Overlays a grid, and selects one point in each grid cell (if one exists). If use_random_centers is True,
#     the point selected is random within that cell. Otherwise it's whichever voxel_grid returns first.
#     """
#     # Voxel grid returns a list, same length as the original, with a mapping to unique grid identifiers
#     xyz_grid_indices = voxel_grid(unbatched_xyz, voxel_size, unbatched_batch_ids)

#     # We wish to take one of each grid identifier, to use as our point
#     # Based on: https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
#     grid_ids, xyz_to_grid_id, grid_cell_counts = torch.unique(
#         xyz_grid_indices, sorted=True, return_inverse=True, return_counts=True
#     )

#     # The grid ids are sorted, and the counts match that sorting. So if we sort xyz_to_grid_id and get the mapping
#     # from that operation, we can use the count maps as indices into the original xyzs, by doing cumsum
#     _, xyz_to_grid_id_sort_mapping = torch.sort(xyz_to_grid_id, stable=True)

#     if use_random_centers:
#         random_offsets = (
#             torch.rand(grid_cell_counts.shape[0]).to(grid_cell_counts.device) * grid_cell_counts
#         ).int()
#     else:
#         random_offsets = 0

#     unique_grid_indices = (
#         torch.cat(
#             (
#                 torch.tensor([0]).to(grid_cell_counts.device),
#                 grid_cell_counts.cumsum(0)[:-1],
#             ),
#             dim=0,
#         )
#         + random_offsets
#     )
#     unique_grid_xyz_indices = xyz_to_grid_id_sort_mapping[unique_grid_indices]

#     # We return the indices into the original data, for consistency with fps
#     return unique_grid_xyz_indices


def get_bounds(points: Tensor, tol: float = 1e-4):
    """Returns min and max along each dimension

    Args:
        points (Tensor): [N, 3]

    Returns:
        mins_and_maxes: [3, 2]
    """
    assert points.ndim == 2 and points.shape[-1] == 3, points.shape
    assert points.shape[0] > 1, f"Points is of shape {points.shape}"
    return torch.stack([points.min(dim=0).values, points.max(dim=0).values], dim=-1)