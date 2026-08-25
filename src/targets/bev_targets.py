import numpy as np

from src.config.constants import (
    BEV_SHAPE,
    POINT_CLOUD_RANGE,
    VOXEL_SIZE
)


def create_bev_targets(boxes):

    x_min, x_max = POINT_CLOUD_RANGE["x"]
    y_min, y_max = POINT_CLOUD_RANGE["y"]

    voxel_x = VOXEL_SIZE["x"]
    voxel_y = VOXEL_SIZE["y"]

    grid_x, grid_y = BEV_SHAPE

    # ------------------------------------
    # Target maps
    # ------------------------------------

    heatmap = np.zeros(
        (grid_x, grid_y),
        dtype=np.float32
    )

    offset = np.zeros(
        (2, grid_x, grid_y),
        dtype=np.float32
    )

    size = np.zeros(
        (2, grid_x, grid_y),
        dtype=np.float32
    )

    rotation = np.zeros(
        (2, grid_x, grid_y),
        dtype=np.float32
    )

    regression_mask = np.zeros(
        (grid_x, grid_y),
        dtype=np.bool_
    )

    for box in boxes:

        center_x = box["center_x"]
        center_y = box["center_y"]

        length = box["length"]
        width = box["width"]

        yaw = box["yaw"]

        # --------------------------------
        # Metric → continuous BEV position
        # --------------------------------

        gx = (
            center_x - x_min
        ) / voxel_x

        gy = (
            center_y - y_min
        ) / voxel_y

        # Integer target cell
        ix = int(np.floor(gx))
        iy = int(np.floor(gy))

        # Safety check
        if not (
            0 <= ix < grid_x and
            0 <= iy < grid_y
        ):
            continue

        # --------------------------------
        # Object center
        # --------------------------------

        heatmap[ix, iy] = 1.0
        regression_mask[ix, iy] = True

        # --------------------------------
        # Sub-cell center offset
        # --------------------------------

        offset[0, ix, iy] = gx - ix
        offset[1, ix, iy] = gy - iy

        # --------------------------------
        # Physical dimensions
        # --------------------------------

        size[0, ix, iy] = length
        size[1, ix, iy] = width

        # --------------------------------
        # Orientation
        # --------------------------------

        rotation[0, ix, iy] = np.sin(yaw)
        rotation[1, ix, iy] = np.cos(yaw)

    return {
        "heatmap": heatmap,
        "offset": offset,
        "size": size,
        "rotation": rotation,
        "regression_mask": regression_mask,
    }
