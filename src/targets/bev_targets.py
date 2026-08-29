"""Center-based BEV target generation."""

import math

import numpy as np

from src.config.constants import BEV_SHAPE, POINT_CLOUD_RANGE, VOXEL_SIZE


def gaussian_2d(shape, sigma=1.0):
    if len(shape) != 2 or any(int(value) != value or value <= 0 for value in shape):
        raise ValueError("shape must contain two positive integers")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    height, width = map(int, shape)
    y, x = np.ogrid[
        -(height - 1) / 2 : (height - 1) / 2 + 1,
        -(width - 1) / 2 : (width - 1) / 2 + 1,
    ]
    return np.exp(-(x * x + y * y) / (2 * sigma * sigma)).astype(np.float32)


def gaussian_radius(height, width, min_overlap=0.7):
    """Return the CenterNet radius for an object footprint measured in cells."""
    values = np.asarray([height, width, min_overlap], dtype=np.float64)
    if not np.isfinite(values).all() or height <= 0 or width <= 0:
        raise ValueError("height and width must be finite and positive")
    if not 0 < min_overlap < 1:
        raise ValueError("min_overlap must lie in (0, 1)")

    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))) / 2

    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))) / 2

    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))) / (2 * a3)
    return max(0.0, min(r1, r2, r3))


def draw_gaussian_xy(heatmap, center_x, center_y, radius):
    heatmap = np.asarray(heatmap)
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be two-dimensional")
    if not isinstance(center_x, (int, np.integer)) or not isinstance(center_y, (int, np.integer)):
        raise TypeError("Gaussian center indices must be integers")
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    grid_x, grid_y = heatmap.shape
    if not (0 <= center_x < grid_x and 0 <= center_y < grid_y):
        raise ValueError("Gaussian center must lie inside the heatmap")

    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    x0, x1 = max(0, center_x - radius), min(grid_x, center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(grid_y, center_y + radius + 1)
    gx0, gy0 = x0 - (center_x - radius), y0 - (center_y - radius)
    patch = gaussian[gx0 : gx0 + (x1 - x0), gy0 : gy0 + (y1 - y0)]
    np.maximum(heatmap[x0:x1, y0:y1], patch, out=heatmap[x0:x1, y0:y1])


def _validated_box(box):
    required = ("center_x", "center_y", "length", "width", "yaw")
    if not isinstance(box, dict) or any(key not in box for key in required):
        raise ValueError(f"each box must contain {required}")
    values = np.asarray([box[key] for key in required], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("box values must be finite")
    if box["length"] <= 0 or box["width"] <= 0:
        raise ValueError("box length and width must be positive")
    return values


def create_bev_targets(boxes):
    if boxes is None:
        raise TypeError("boxes must be an iterable, not None")

    grid_x, grid_y = BEV_SHAPE
    heatmap = np.zeros((grid_x, grid_y), dtype=np.float32)
    offset = np.zeros((2, grid_x, grid_y), dtype=np.float32)
    size = np.zeros((2, grid_x, grid_y), dtype=np.float32)
    rotation = np.zeros((2, grid_x, grid_y), dtype=np.float32)
    regression_mask = np.zeros((grid_x, grid_y), dtype=np.float32)
    x_min, _ = POINT_CLOUD_RANGE["x"]
    y_min, _ = POINT_CLOUD_RANGE["y"]
    voxel_x, voxel_y = VOXEL_SIZE["x"], VOXEL_SIZE["y"]

    for box in boxes:
        center_x, center_y, length, width, yaw = _validated_box(box)
        gx, gy = (center_x - x_min) / voxel_x, (center_y - y_min) / voxel_y
        ix, iy = int(np.floor(gx)), int(np.floor(gy))
        if not (0 <= ix < grid_x and 0 <= iy < grid_y):
            continue

        footprint_x = length / voxel_x
        footprint_y = width / voxel_y
        radius = max(1, int(gaussian_radius(footprint_x, footprint_y)))
        draw_gaussian_xy(heatmap, ix, iy, radius)
        if regression_mask[ix, iy] and length * width <= size[0, ix, iy] * size[1, ix, iy]:
            continue
        offset[:, ix, iy] = (gx - ix, gy - iy)
        size[:, ix, iy] = (length, width)
        rotation[:, ix, iy] = (np.sin(yaw), np.cos(yaw))
        regression_mask[ix, iy] = 1.0

    return {
        "heatmap": heatmap,
        "offset": offset,
        "size": size,
        "rotation": rotation,
        "regression_mask": regression_mask,
    }
