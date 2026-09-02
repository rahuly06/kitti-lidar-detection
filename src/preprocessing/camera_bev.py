"""Aggregate camera colours attached to LiDAR points into a BEV tensor."""

import numpy as np

from src.config.constants import BEV_SHAPE, POINT_CLOUD_RANGE, VOXEL_SIZE


def aggregate_rgb_points_to_bev(fused_points: np.ndarray) -> np.ndarray:
    """Create mean-RGB and visibility BEV channels from fused points.

    ``fused_points`` must contain ``x, y, z, intensity, r, g, b`` columns.
    RGB values are expected in ``[0, 1]``. The result has red, green, blue,
    and binary camera-visibility channels with shape ``(4, *BEV_SHAPE)``.
    """
    fused_points = np.asarray(fused_points)
    if fused_points.ndim != 2 or fused_points.shape[1] != 7:
        raise ValueError(
            f"fused_points must have shape (N, 7), got {fused_points.shape}"
        )
    if not np.issubdtype(fused_points.dtype, np.number):
        raise TypeError("fused_points must contain numeric values")

    x_min, x_max = POINT_CLOUD_RANGE["x"]
    y_min, y_max = POINT_CLOUD_RANGE["y"]
    finite = np.isfinite(fused_points).all(axis=1)
    inside = (
        finite
        & (fused_points[:, 0] >= x_min)
        & (fused_points[:, 0] < x_max)
        & (fused_points[:, 1] >= y_min)
        & (fused_points[:, 1] < y_max)
    )
    points = fused_points[inside].astype(np.float32, copy=False)

    rgb_sums = np.zeros((3, *BEV_SHAPE), dtype=np.float32)
    point_count = np.zeros(BEV_SHAPE, dtype=np.float32)
    if points.size == 0:
        return np.concatenate((rgb_sums, point_count[None]), axis=0)

    rgb = points[:, 4:7]
    if ((rgb < 0.0) | (rgb > 1.0)).any():
        raise ValueError("RGB values must be normalized to [0, 1]")

    x_idx = ((points[:, 0] - x_min) / VOXEL_SIZE["x"]).astype(np.int32)
    y_idx = ((points[:, 1] - y_min) / VOXEL_SIZE["y"]).astype(np.int32)
    for channel in range(3):
        np.add.at(rgb_sums[channel], (x_idx, y_idx), rgb[:, channel])
    np.add.at(point_count, (x_idx, y_idx), 1.0)

    visible = point_count > 0
    rgb_sums[:, visible] /= point_count[visible]
    visibility = visible.astype(np.float32)
    return np.concatenate((rgb_sums, visibility[None]), axis=0)
