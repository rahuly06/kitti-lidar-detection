"""Reusable inference and lightweight evaluation helpers for BEV detection."""

import math
from pathlib import Path

import numpy as np
import torch

from src.geometry.boxes import get_lidar_boxes
from src.inference.decode import decode_predictions
from src.models.bev_detector import BEVDetector
from src.preprocessing.bev import bev_projection
from src.preprocessing.fusion import build_fused_bev

INPUT_CHANNELS = {"lidar": 3, "fusion": 7}


def resolve_inference_device(device="auto"):
    """Resolve ``auto``, a device string, or an existing torch device."""
    if isinstance(device, torch.device):
        resolved = device
    elif device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def load_detector_checkpoint(checkpoint_path, device="auto"):
    """Load a detector and return ``(model, checkpoint, resolved_device)``."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    resolved_device = resolve_inference_device(device)
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain model_state_dict")
    config = checkpoint.setdefault("config", {})
    input_mode = config.get("input_mode")
    input_channels = config.get("input_channels")
    if input_channels is None:
        stem_weight = checkpoint["model_state_dict"].get("stem.0.weight")
        if stem_weight is None or stem_weight.ndim != 4:
            raise ValueError("checkpoint does not identify its model input channels")
        input_channels = int(stem_weight.shape[1])
    if input_mode is None:
        modes = [mode for mode, channels in INPUT_CHANNELS.items() if channels == input_channels]
        if len(modes) != 1:
            raise ValueError("checkpoint input mode cannot be inferred from its weights")
        input_mode = modes[0]
    if input_mode not in INPUT_CHANNELS:
        raise ValueError("checkpoint input_mode must be 'lidar' or 'fusion'")
    expected_channels = INPUT_CHANNELS[input_mode]
    if input_channels != expected_channels:
        raise ValueError(
            f"{input_mode} checkpoint must use {expected_channels} channels, "
            f"got {input_channels}"
        )
    config.setdefault("input_mode", input_mode)
    config.setdefault("input_channels", input_channels)
    model = BEVDetector(input_channels=input_channels).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, resolved_device


def build_model_input(sample, input_mode):
    """Build the same BEV representation used to train the selected model."""
    if input_mode == "lidar":
        bev = bev_projection(sample["points"])
    elif input_mode == "fusion":
        if "image" not in sample:
            raise KeyError("fusion inference requires sample['image']")
        bev = build_fused_bev(sample["points"], sample["image"], sample["calib"])
    else:
        raise ValueError("input_mode must be 'lidar' or 'fusion'")
    return np.asarray(bev, dtype=np.float32)


def infer_sample(
    model, sample, input_mode, device="auto", score_threshold=0.3,
    top_k=50, min_center_distance=1.5,
):
    """Run one KITTI sample through preprocessing, the model, and decoding."""
    resolved_device = resolve_inference_device(device)
    bev = build_model_input(sample, input_mode)
    inputs = torch.from_numpy(bev).unsqueeze(0).to(resolved_device)
    with torch.inference_mode():
        outputs = model(inputs)
    detections = decode_predictions(
        outputs, score_threshold=score_threshold, top_k=top_k,
        min_center_distance=min_center_distance,
    )
    return {
        "id": sample["id"], "bev": bev, "detections": detections,
        "maximum_score": float(torch.sigmoid(outputs["heatmap"]).max().item()),
    }


def match_detections(detections, ground_truth, match_distance=2.0):
    """Greedily match score-ordered predictions to GT centers within a radius."""
    if not math.isfinite(match_distance) or match_distance <= 0:
        raise ValueError("match_distance must be finite and positive")
    unmatched = set(range(len(ground_truth)))
    matches, false_positive_indices = [], []
    for detection_index, detection in enumerate(detections):
        candidates = [(
            math.hypot(
                detection["center_x"] - ground_truth[index]["center_x"],
                detection["center_y"] - ground_truth[index]["center_y"],
            ), index,
        ) for index in unmatched]
        distance, gt_index = min(candidates, default=(math.inf, None))
        if distance <= match_distance:
            unmatched.remove(gt_index)
            matches.append((detection_index, gt_index, distance))
        else:
            false_positive_indices.append(detection_index)
    return {
        "matches": matches,
        "false_positive_indices": false_positive_indices,
        "false_negative_indices": sorted(unmatched),
    }


def evaluate_dataset(
    model, dataset, indices, input_mode, device="auto", score_threshold=0.3,
    top_k=50, min_center_distance=1.5, match_distance=2.0, retain_bev=False,
):
    """Evaluate labeled samples using the project's center-distance diagnostic."""
    if not getattr(dataset, "has_labels", False):
        raise ValueError("evaluate_dataset requires a labeled dataset")
    records, center_errors = [], []
    true_positives = false_positives = false_negatives = 0
    for sample_index in indices:
        sample = dataset[sample_index]
        result = infer_sample(
            model, sample, input_mode, device, score_threshold, top_k,
            min_center_distance,
        )
        ground_truth = get_lidar_boxes(sample["labels"], sample["calib"])
        matching = match_detections(result["detections"], ground_truth, match_distance)
        true_positives += len(matching["matches"])
        false_positives += len(matching["false_positive_indices"])
        false_negatives += len(matching["false_negative_indices"])
        center_errors.extend(match[2] for match in matching["matches"])
        if not retain_bev:
            result = {key: value for key, value in result.items() if key != "bev"}
        records.append({**result, "index": sample_index, "ground_truth": ground_truth,
                        "matching": matching})
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    metrics = {
        "samples": len(records), "true_positives": true_positives,
        "false_positives": false_positives, "false_negatives": false_negatives,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_center_error": float(np.mean(center_errors)) if center_errors else math.nan,
    }
    return metrics, records


def infer_dataset(
    model, dataset, indices, input_mode, device="auto", retain_bev=False, **decode_options
):
    """Run inference on labeled or unlabeled samples without computing metrics."""
    records = []
    for index in indices:
        result = infer_sample(
            model, dataset[index], input_mode, device=device, **decode_options
        )
        if not retain_bev:
            result = {key: value for key, value in result.items() if key != "bev"}
        records.append({**result, "index": index})
    return records


def summarize_predictions(records):
    """Summarize predictions where ground truth is unavailable."""
    counts = np.asarray([len(record["detections"]) for record in records], dtype=float)
    scores = [d["score"] for record in records for d in record["detections"]]
    return {
        "samples": len(records),
        "total_detections": int(counts.sum()) if counts.size else 0,
        "frames_with_detections": int(np.count_nonzero(counts)) if counts.size else 0,
        "mean_detections_per_frame": float(counts.mean()) if counts.size else 0.0,
        "mean_score": float(np.mean(scores)) if scores else math.nan,
    }


