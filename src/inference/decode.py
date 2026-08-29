"""Decode dense BEV detector outputs into metric-space boxes."""

import math

import torch
import torch.nn.functional as F

from src.config.constants import POINT_CLOUD_RANGE, VOXEL_SIZE


_REQUIRED_KEYS = ("heatmap", "offset", "size", "rotation")


def decode_predictions(outputs, score_threshold=0.5, top_k=50, min_center_distance=1.5):
    """Decode one batch element and suppress duplicate nearby centers."""
    if not isinstance(outputs, dict):
        raise TypeError("outputs must be a dictionary")
    missing = set(_REQUIRED_KEYS).difference(outputs)
    if missing:
        raise KeyError(f"missing output keys: {sorted(missing)}")
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must lie in [0, 1]")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(min_center_distance) or min_center_distance < 0:
        raise ValueError("min_center_distance must be finite and non-negative")

    heatmap_logits = outputs["heatmap"]
    if heatmap_logits.ndim != 4 or heatmap_logits.shape[:2] != (1, 1):
        raise ValueError("heatmap must have shape (1, 1, H, W)")
    height, width = heatmap_logits.shape[2:]
    expected_shapes = {
        "offset": (1, 2, height, width),
        "size": (1, 2, height, width),
        "rotation": (1, 2, height, width),
    }
    for key, expected in expected_shapes.items():
        if tuple(outputs[key].shape) != expected:
            raise ValueError(f"{key} must have shape {expected}, got {tuple(outputs[key].shape)}")
    if not all(torch.isfinite(outputs[key]).all() for key in _REQUIRED_KEYS):
        raise ValueError("model outputs must be finite")

    heatmap = torch.sigmoid(heatmap_logits[0, 0])
    pooled = F.max_pool2d(heatmap[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    candidates = torch.where(heatmap == pooled, heatmap, torch.zeros_like(heatmap))
    scores, indices = torch.topk(candidates.flatten(), min(top_k, candidates.numel()))

    offset, size, rotation = (outputs[key][0] for key in ("offset", "size", "rotation"))
    x_min, _ = POINT_CLOUD_RANGE["x"]
    y_min, _ = POINT_CLOUD_RANGE["y"]
    decoded = []
    for score_tensor, flat_index_tensor in zip(scores, indices):
        score = float(score_tensor.item())
        if score < score_threshold:
            continue
        ix, iy = divmod(int(flat_index_tensor.item()), width)
        length, box_width = (float(value) for value in size[:, ix, iy])
        if length <= 0 or box_width <= 0:
            continue
        dx, dy = (float(value) for value in offset[:, ix, iy])
        sin_yaw, cos_yaw = (float(value) for value in rotation[:, ix, iy])
        decoded.append({
            "score": score,
            "center_x": (ix + dx) * VOXEL_SIZE["x"] + x_min,
            "center_y": (iy + dy) * VOXEL_SIZE["y"] + y_min,
            "length": length,
            "width": box_width,
            "yaw": math.atan2(sin_yaw, cos_yaw),
        })

    kept = []
    for detection in decoded:
        duplicate = any(
            math.hypot(
                detection["center_x"] - previous["center_x"],
                detection["center_y"] - previous["center_y"],
            ) < min_center_distance
            for previous in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept
