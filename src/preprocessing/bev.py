"""Bird's-eye-view representation for KITTI LiDAR point clouds."""

import numpy as np

from src.config.constants import BEV_SHAPE, POINT_CLOUD_RANGE, VOXEL_SIZE


def bev_projection(points: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, intensity]`` points to a three-channel BEV tensor."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"points must have shape (N, 4), got {points.shape}")
    if not np.issubdtype(points.dtype, np.number):
        raise TypeError("points must contain numeric values")

    x_min, x_max = POINT_CLOUD_RANGE["x"]
    y_min, y_max = POINT_CLOUD_RANGE["y"]
    z_min, z_max = POINT_CLOUD_RANGE["z"]

    finite = np.isfinite(points).all(axis=1)
    inside = (
        finite
        & (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] < z_max)
    )
    points = points[inside].astype(np.float32, copy=False)

    height = np.zeros(BEV_SHAPE, dtype=np.float32)
    density = np.zeros(BEV_SHAPE, dtype=np.float32)
    intensity = np.zeros(BEV_SHAPE, dtype=np.float32)
    if points.size == 0:
        return np.stack((height, density, intensity), axis=0)

    x_idx = ((points[:, 0] - x_min) / VOXEL_SIZE["x"]).astype(np.int32)
    y_idx = ((points[:, 1] - y_min) / VOXEL_SIZE["y"]).astype(np.int32)
    normalized_height = (points[:, 2] - z_min) / (z_max - z_min)

    np.maximum.at(height, (x_idx, y_idx), normalized_height)
    np.add.at(density, (x_idx, y_idx), 1.0)
    np.maximum.at(intensity, (x_idx, y_idx), points[:, 3])

    np.minimum(density, np.float32(64.0), out=density)
    np.log1p(density, out=density)
    density /= np.float32(np.log(65.0))
    np.clip(intensity, 0.0, 1.0, out=intensity)

    return np.stack((height, density, intensity), axis=0)
