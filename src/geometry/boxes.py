import numpy as np

from src.config.constants import POINT_CLOUD_RANGE
from src.geometry.transforms import lidar_xy_to_bev, rectified_camera_to_lidar


def create_box_corners_camera(dimensions, location, rotation_y):
    """
    Create the 8 corners of a KITTI 3D bounding box
    in rectified camera coordinates.

    Parameters
    ----------
    dimensions : array-like
        [height, width, length]

    location : array-like
        [x, y, z] bottom-center of the KITTI box

    rotation_y : float
        Rotation around the camera Y-axis

    Returns
    -------
    corners : np.ndarray
        Shape (8, 3)
    """

    h, w, l = dimensions
    x, y, z = location

    # Box centered around the object's bottom-center
    x_corners = np.array([
         l / 2,  l / 2, -l / 2, -l / 2,
         l / 2,  l / 2, -l / 2, -l / 2
    ])

    y_corners = np.array([
         0,  0,  0,  0,
        -h, -h, -h, -h
    ])

    z_corners = np.array([
         w / 2, -w / 2, -w / 2,  w / 2,
         w / 2, -w / 2, -w / 2,  w / 2
    ])

    corners = np.vstack([
        x_corners,
        y_corners,
        z_corners
    ])

    # Rotate around camera Y axis
    c = np.cos(rotation_y)
    s = np.sin(rotation_y)

    rotation = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

    corners = rotation @ corners

    # Translate to object's KITTI location
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z

    return corners.T

def get_bev_boxes(labels, calib):
    bev_boxes = []

    for obj in labels:

        # Initially only detect cars
        if obj["type"] != "Car":
            continue

        # -------------------------------------
        # Create 3D camera box
        # -------------------------------------

        corners_camera = create_box_corners_camera(
            obj["dimensions"],
            obj["location"],
            obj["rotation_y"]
        )

        # -------------------------------------
        # Camera -> LiDAR
        # -------------------------------------

        corners_lidar = rectified_camera_to_lidar(
            corners_camera,
            calib["tr_velo_to_cam"],
            calib["r0_rect"]
        )

        # -------------------------------------
        # Keep bottom XY corners
        # -------------------------------------

        bottom_xy = corners_lidar[:4, :2]

        # -------------------------------------
        # Check whether box is in detection ROI
        # -------------------------------------

        center = bottom_xy.mean(axis=0)

        x_min, x_max = POINT_CLOUD_RANGE["x"]
        y_min, y_max = POINT_CLOUD_RANGE["y"]

        if not (
            x_min <= center[0] < x_max and
            y_min <= center[1] < y_max
        ):
            continue

        # -------------------------------------
        # Convert to BEV
        # -------------------------------------

        corners_bev = lidar_xy_to_bev(
            bottom_xy
        )

        bev_boxes.append({
            "type": obj["type"],
            "corners_lidar": bottom_xy,
            "corners_bev": corners_bev
        })

    return bev_boxes

def box_parameters_from_lidar_corners(corners):
    """
    Convert four BEV box corners in LiDAR coordinates
    into center, length, width, and yaw.

    corners: shape (4, 2)
    """

    center = corners.mean(axis=0)

    edge_01 = corners[1] - corners[0]
    edge_12 = corners[2] - corners[1]

    width = np.linalg.norm(edge_01)
    length = np.linalg.norm(edge_12)

    # Edge 1 -> 2 points opposite the labelled positive length direction.
    length_vector = -edge_12
    yaw = np.arctan2(length_vector[1], length_vector[0])
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    return {
        "center_x": center[0],
        "center_y": center[1],
        "length": length,
        "width": width,
        "yaw": yaw
    }

def get_lidar_boxes(labels, calib):
    boxes = []

    for obj in labels:

        # For now our detector only learns Cars
        if obj["type"] != "Car":
            continue

        # 1. KITTI label -> 8 corners in camera coordinates
        corners_camera = create_box_corners_camera(
            obj["dimensions"],
            obj["location"],
            obj["rotation_y"]
        )

        # 2. Camera coordinates -> LiDAR coordinates
        corners_lidar = rectified_camera_to_lidar(
            corners_camera,
            calib["tr_velo_to_cam"],
            calib["r0_rect"]
        )

        # 3. Bottom four corners -> BEV footprint
        bottom_xy = corners_lidar[:4, :2]

        # 4. Corners -> center, length, width, yaw
        params = box_parameters_from_lidar_corners(
            bottom_xy
        )

        # 5. Keep only cars inside our BEV region
        x_min, x_max = POINT_CLOUD_RANGE["x"]
        y_min, y_max = POINT_CLOUD_RANGE["y"]
        if not (
            x_min <= params["center_x"] < x_max and
            y_min <= params["center_y"] < y_max
        ):
            continue

        boxes.append(params)

    return boxes
