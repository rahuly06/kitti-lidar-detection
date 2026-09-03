"""Build model-ready BEV features from aligned KITTI LiDAR and camera data."""

import numpy as np

from src.config.constants import BEV_SHAPE
from src.geometry.camera_projection import project_lidar_to_image
from src.preprocessing.bev import bev_projection
from src.preprocessing.camera_bev import aggregate_rgb_points_to_bev


FUSED_BEV_CHANNELS = (
    "lidar_height",
    "lidar_density",
    "lidar_intensity",
    "camera_red",
    "camera_green",
    "camera_blue",
    "camera_visibility",
)


def _validate_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("image must contain numeric values")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    if image.size and (image.min() < 0 or image.max() > 255):
        raise ValueError("image values must lie in [0, 255]")
    return image


def paint_lidar_points_with_rgb(points, image, calibration):
    """Attach normalized camera RGB values to visible LiDAR points.

    Returns (painted_points, pixels, depths). Painted points have columns
    x, y, z, intensity, red, green, blue. Pixels and depths remain available
    for projection diagnostics and future occlusion filtering.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"points must have shape (N, 4), got {points.shape}")
    if not np.issubdtype(points.dtype, np.number):
        raise TypeError("points must contain numeric values")
    image = _validate_image(image)

    pixels, valid_indices, depths = project_lidar_to_image(
        points=points,
        calibration=calibration,
        image_shape=image.shape,
    )
    visible_points = points[valid_indices].astype(np.float32, copy=False)

    pixel_u = np.rint(pixels[:, 0]).astype(np.int64)
    pixel_v = np.rint(pixels[:, 1]).astype(np.int64)
    pixel_u = np.clip(pixel_u, 0, image.shape[1] - 1)
    pixel_v = np.clip(pixel_v, 0, image.shape[0] - 1)

    rgb = image[pixel_v, pixel_u].astype(np.float32) / np.float32(255.0)
    painted_points = np.concatenate((visible_points, rgb), axis=1)
    return painted_points, pixels, depths


def build_fused_bev(points, image, calibration):
    """Return aligned LiDAR and camera features with shape (7, H, W)."""
    lidar_bev = bev_projection(points)
    painted_points, _, _ = paint_lidar_points_with_rgb(
        points,
        image,
        calibration,
    )
    camera_bev = aggregate_rgb_points_to_bev(painted_points)
    fused_bev = np.concatenate((lidar_bev, camera_bev), axis=0).astype(
        np.float32,
        copy=False,
    )

    expected_shape = (len(FUSED_BEV_CHANNELS), *BEV_SHAPE)
    if fused_bev.shape != expected_shape:
        raise RuntimeError(
            f"fused BEV must have shape {expected_shape}, got {fused_bev.shape}"
        )
    if not np.isfinite(fused_bev).all():
        raise RuntimeError("fused BEV contains non-finite values")
    return fused_bev
