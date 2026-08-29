"""Coordinate transforms used by the KITTI BEV pipeline."""

import numpy as np

from src.config.constants import POINT_CLOUD_RANGE, VOXEL_SIZE


def _numeric_array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float64, copy=False)


def rectified_camera_to_lidar(points_rect, tr_velo_to_cam, r0_rect):
    """Transform (N, 3) rectified-camera points into LiDAR coordinates."""
    points_rect = np.asarray(points_rect)
    if points_rect.ndim != 2 or points_rect.shape[1] != 3:
        raise ValueError(f"points_rect must have shape (N, 3), got {points_rect.shape}")
    if not np.issubdtype(points_rect.dtype, np.number):
        raise TypeError("points_rect must contain numeric values")
    if not np.isfinite(points_rect).all():
        raise ValueError("points_rect must contain only finite values")
    points_rect = points_rect.astype(np.float64, copy=False)
    tr_velo_to_cam = _numeric_array(tr_velo_to_cam, (3, 4), "tr_velo_to_cam")
    r0_rect = _numeric_array(r0_rect, (3, 3), "r0_rect")

    try:
        points_camera = (np.linalg.inv(r0_rect) @ points_rect.T).T
        transform = np.eye(4, dtype=np.float64)
        transform[:3] = tr_velo_to_cam
        camera_to_lidar = np.linalg.inv(transform)
    except np.linalg.LinAlgError as exc:
        raise ValueError("calibration transforms must be invertible") from exc

    homogeneous = np.column_stack((points_camera, np.ones(points_camera.shape[0])))
    return (camera_to_lidar @ homogeneous.T).T[:, :3]


def lidar_xy_to_bev(xy):
    """Convert (N, 2) LiDAR XY coordinates to continuous BEV indices."""
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (N, 2), got {xy.shape}")
    if not np.issubdtype(xy.dtype, np.number):
        raise TypeError("xy must contain numeric values")
    if not np.isfinite(xy).all():
        raise ValueError("xy must contain only finite values")

    x_min, _ = POINT_CLOUD_RANGE["x"]
    y_min, _ = POINT_CLOUD_RANGE["y"]
    return np.column_stack(
        ((xy[:, 0] - x_min) / VOXEL_SIZE["x"], (xy[:, 1] - y_min) / VOXEL_SIZE["y"])
    )
