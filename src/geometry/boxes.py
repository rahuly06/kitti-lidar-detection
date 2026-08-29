"""3D box conversion helpers for KITTI labels."""

import numpy as np

from src.config.constants import POINT_CLOUD_RANGE
from src.geometry.transforms import lidar_xy_to_bev, rectified_camera_to_lidar


def _vector(value, length, name):
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float64, copy=False)


def create_box_corners_camera(dimensions, location, rotation_y):
    """Create eight KITTI box corners in rectified camera coordinates."""
    dimensions = _vector(dimensions, 3, "dimensions")
    location = _vector(location, 3, "location")
    if np.any(dimensions <= 0):
        raise ValueError("box dimensions must be positive")
    if not np.isfinite(rotation_y):
        raise ValueError("rotation_y must be finite")

    height, width, length = dimensions
    x, y, z = location
    corners = np.array(
        [
            [length / 2, 0, width / 2],
            [length / 2, 0, -width / 2],
            [-length / 2, 0, -width / 2],
            [-length / 2, 0, width / 2],
            [length / 2, -height, width / 2],
            [length / 2, -height, -width / 2],
            [-length / 2, -height, -width / 2],
            [-length / 2, -height, width / 2],
        ],
        dtype=np.float64,
    )
    cosine, sine = np.cos(rotation_y), np.sin(rotation_y)
    rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    return (rotation @ corners.T).T + np.array([x, y, z])


def box_parameters_from_lidar_corners(corners):
    """Convert four ordered LiDAR XY corners into center, size, and yaw."""
    corners = np.asarray(corners)
    if corners.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {corners.shape}")
    if not np.issubdtype(corners.dtype, np.number):
        raise TypeError("corners must contain numeric values")
    if not np.isfinite(corners).all():
        raise ValueError("corners must contain only finite values")

    center = corners.mean(axis=0)
    width = np.linalg.norm(corners[1] - corners[0])
    length_vector = corners[1] - corners[2]
    length = np.linalg.norm(length_vector)
    if length <= 0 or width <= 0:
        raise ValueError("corners must describe a box with positive length and width")
    yaw = np.arctan2(length_vector[1], length_vector[0])
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "length": float(length),
        "width": float(width),
        "yaw": float(yaw),
    }


def box_corners_from_parameters(center_x, center_y, length, width, yaw):
    """Return four ordered BEV corners in LiDAR XY coordinates."""
    values = np.asarray([center_x, center_y, length, width, yaw], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("box parameters must be finite")
    if length <= 0 or width <= 0:
        raise ValueError("length and width must be positive")
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ]
    )
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return (rotation @ local.T).T + np.array([center_x, center_y])


def _lidar_box_parameters(obj, calib):
    required = {"type", "dimensions", "location", "rotation_y"}
    if not isinstance(obj, dict) or not required.issubset(obj):
        raise ValueError(f"label must contain {sorted(required)}")
    if not isinstance(calib, dict) or not {"tr_velo_to_cam", "r0_rect"}.issubset(calib):
        raise ValueError("calib must contain tr_velo_to_cam and r0_rect")
    corners_camera = create_box_corners_camera(
        obj["dimensions"], obj["location"], obj["rotation_y"]
    )
    corners_lidar = rectified_camera_to_lidar(
        corners_camera, calib["tr_velo_to_cam"], calib["r0_rect"]
    )
    bottom_xy = corners_lidar[:4, :2]
    return bottom_xy, box_parameters_from_lidar_corners(bottom_xy)


def _inside_roi(params):
    x_min, x_max = POINT_CLOUD_RANGE["x"]
    y_min, y_max = POINT_CLOUD_RANGE["y"]
    return x_min <= params["center_x"] < x_max and y_min <= params["center_y"] < y_max


def get_bev_boxes(labels, calib):
    """Return LiDAR and BEV corners for Car labels whose centers are in range."""
    boxes = []
    for obj in labels:
        if not isinstance(obj, dict) or "type" not in obj:
            raise ValueError("each label must be a dictionary containing type")
        if obj["type"] != "Car":
            continue
        bottom_xy, params = _lidar_box_parameters(obj, calib)
        if _inside_roi(params):
            boxes.append(
                {
                    "type": "Car",
                    "corners_lidar": bottom_xy,
                    "corners_bev": lidar_xy_to_bev(bottom_xy),
                }
            )
    return boxes


def get_lidar_boxes(labels, calib):
    """Return parameterized LiDAR boxes for in-range Car labels."""
    boxes = []
    for obj in labels:
        if not isinstance(obj, dict) or "type" not in obj:
            raise ValueError("each label must be a dictionary containing type")
        if obj["type"] != "Car":
            continue
        _, params = _lidar_box_parameters(obj, calib)
        if _inside_roi(params):
            boxes.append(params)
    return boxes
