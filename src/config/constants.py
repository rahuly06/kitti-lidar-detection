POINT_CLOUD_RANGE = {
    "x": (0.0, 70.0),
    "y": (-40.0, 40.0),
    "z": (-3.0, 1.0),
}

VOXEL_SIZE = {
    "x": 0.2,
    "y": 0.2,
    "z": 0.2,
}

BEV_SHAPE = tuple(
    int((POINT_CLOUD_RANGE[axis][1] - POINT_CLOUD_RANGE[axis][0]) / VOXEL_SIZE[axis])
    for axis in ("x", "y")
)
