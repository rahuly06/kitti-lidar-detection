import numpy as np
from src.config.constants import POINT_CLOUD_RANGE, VOXEL_SIZE


def rectified_camera_to_lidar(
    points_rect,
    tr_velo_to_cam,
    r0_rect
):
    """
    Transform Nx3 points from rectified camera coordinates
    into LiDAR coordinates.
    """

    # ---------------------------------------
    # Rectified camera -> unrectified camera
    # ---------------------------------------

    R_rect_inv = np.linalg.inv(r0_rect)

    points_camera = (
        R_rect_inv @ points_rect.T
    ).T

    # ---------------------------------------
    # Convert Tr_velo_to_cam to 4x4
    # ---------------------------------------

    T_velo_cam = np.eye(4, dtype=np.float32)

    T_velo_cam[:3, :] = tr_velo_to_cam

    # Camera -> LiDAR
    T_cam_velo = np.linalg.inv(T_velo_cam)

    # ---------------------------------------
    # Homogeneous coordinates
    # ---------------------------------------

    points_camera_h = np.hstack([
        points_camera,
        np.ones((points_camera.shape[0], 1))
    ])

    points_lidar_h = (
        T_cam_velo @ points_camera_h.T
    ).T

    return points_lidar_h[:, :3]

def lidar_xy_to_bev(xy):
    """
    Convert LiDAR XY coordinates in meters
    to BEV grid indices.

    xy shape: (N, 2)

    Returns:
        (N, 2) array of [x_index, y_index]
    """

    x_min, x_max = POINT_CLOUD_RANGE["x"]
    y_min, y_max = POINT_CLOUD_RANGE["y"]

    voxel_x = VOXEL_SIZE["x"]
    voxel_y = VOXEL_SIZE["y"]

    x_idx = (
        (xy[:, 0] - x_min) / voxel_x
    )

    y_idx = (
        (xy[:, 1] - y_min) / voxel_y
    )

    return np.column_stack([
        x_idx,
        y_idx
    ])