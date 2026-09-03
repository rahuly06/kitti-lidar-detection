import math

import numpy as np
import pytest
import torch

from src.inference.pipeline import (
    build_model_input, match_detections, resolve_inference_device,
    summarize_predictions,
)


def test_build_lidar_model_input():
    bev = build_model_input({"points": np.empty((0, 4), dtype=np.float32)}, "lidar")
    assert bev.shape == (3, 350, 400)
    assert bev.dtype == np.float32


def test_fusion_input_requires_image():
    with pytest.raises(KeyError, match="image"):
        build_model_input({"points": np.empty((0, 4)), "calib": {}}, "fusion")


def test_match_detections_counts_and_center_error():
    detections = [{"center_x": 10.5, "center_y": 2.0},
                  {"center_x": 40.0, "center_y": 10.0}]
    ground_truth = [{"center_x": 10.0, "center_y": 2.0},
                    {"center_x": 20.0, "center_y": 3.0}]
    result = match_detections(detections, ground_truth, match_distance=2.0)
    assert result["matches"] == [(0, 0, 0.5)]
    assert result["false_positive_indices"] == [1]
    assert result["false_negative_indices"] == [1]


def test_prediction_summary_handles_empty_and_nonempty_records():
    assert math.isnan(summarize_predictions([])["mean_score"])
    summary = summarize_predictions([
        {"detections": [{"score": 0.5}, {"score": 0.7}]}, {"detections": []},
    ])
    assert summary["total_detections"] == 2
    assert summary["frames_with_detections"] == 1
    assert summary["mean_detections_per_frame"] == pytest.approx(1.0)
    assert summary["mean_score"] == pytest.approx(0.6)


def test_device_and_match_distance_validation():
    assert resolve_inference_device(torch.device("cpu")) == torch.device("cpu")
    with pytest.raises(ValueError, match="positive"):
        match_detections([], [], match_distance=0)
