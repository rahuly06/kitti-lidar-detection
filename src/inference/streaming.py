"""Streaming inference and conventional forward-up BEV rendering."""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config.constants import BEV_SHAPE
from src.geometry.boxes import box_corners_from_parameters
from src.geometry.transforms import lidar_xy_to_bev
from src.inference.decode import decode_predictions
from src.inference.pipeline import build_model_input, resolve_inference_device


def lidar_xy_to_stream_pixels(xy, bev_shape=BEV_SHAPE):
    """Map LiDAR XY to pixels with forward up and vehicle-right on screen-right."""
    if len(bev_shape) != 2 or any(int(size) <= 0 for size in bev_shape):
        raise ValueError("bev_shape must contain two positive dimensions")
    indices = lidar_xy_to_bev(xy)
    forward_cells, lateral_cells = (int(size) for size in bev_shape)
    return np.column_stack((
        lateral_cells - 1 - indices[:, 1],
        forward_cells - 1 - indices[:, 0],
    ))


def density_to_bgr(bev):
    """Render LiDAR density in conventional ego-centric BEV orientation."""
    bev = np.asarray(bev)
    if bev.ndim != 3 or bev.shape[0] < 2:
        raise ValueError("bev must have shape (C, H, W) with a density channel")
    density = np.flip(bev[1], axis=(0, 1))
    nonzero = density[density > 0]
    upper = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
    normalized = np.clip(density / max(upper, 1e-6), 0.0, 1.0)
    grayscale = np.rint(normalized * 255).astype(np.uint8)
    return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)


def draw_detection(frame, detection, bev_shape=BEV_SHAPE, draw_heading=True):
    """Draw one orange box and score, with an optional red heading arrow."""
    corners_xy = box_corners_from_parameters(
        detection["center_x"], detection["center_y"],
        detection["length"], detection["width"], detection["yaw"],
    )
    corners = np.rint(lidar_xy_to_stream_pixels(corners_xy, bev_shape)).astype(np.int32)
    cv2.polylines(frame, [corners], True, (0, 165, 255), 2, cv2.LINE_AA)

    center_xy = np.array([[detection["center_x"], detection["center_y"]]])
    center = np.rint(lidar_xy_to_stream_pixels(center_xy, bev_shape)[0]).astype(int)
    cv2.putText(
        frame, f"{detection['score']:.2f}", tuple(center),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA,
    )
    if draw_heading:
        front_xy = center_xy[0] + 0.5 * detection["length"] * np.array([
            np.cos(detection["yaw"]), np.sin(detection["yaw"])
        ])
        endpoints = lidar_xy_to_stream_pixels(
            np.vstack((center_xy[0], front_xy)), bev_shape
        )
        start, end = np.rint(endpoints).astype(int)
        cv2.arrowedLine(
            frame, tuple(start), tuple(end), (0, 0, 255), 2,
            cv2.LINE_AA, tipLength=0.3,
        )


def create_stream_dashboard(
    sample, bev, detections, timings, output_size=(1200, 360), draw_heading=True
):
    """Create a BGR camera-and-BEV dashboard with a conventional BEV view."""
    output_width, output_height = (int(value) for value in output_size)
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output_size dimensions must be positive")
    bev_frame = density_to_bgr(bev)
    bev_shape = tuple(bev.shape[1:])
    for detection in detections:
        draw_detection(bev_frame, detection, bev_shape, draw_heading)

    image = np.asarray(sample["image"])
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("sample image must have shape (H, W, 3)")
    camera_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    target_height = max(bev_frame.shape[0], 400)
    camera_width = round(camera_frame.shape[1] * target_height / camera_frame.shape[0])
    camera_frame = cv2.resize(camera_frame, (camera_width, target_height))
    bev_width = round(bev_frame.shape[1] * target_height / bev_frame.shape[0])
    bev_frame = cv2.resize(
        bev_frame, (bev_width, target_height), interpolation=cv2.INTER_NEAREST
    )
    dashboard = np.hstack((camera_frame, bev_frame))
    text = (
        f"frame {sample['id']} | cars {len(detections)} | "
        f"network {1000 * timings['network']:.1f} ms | "
        f"pipeline {1000 * timings['pipeline']:.1f} ms"
    )
    cv2.rectangle(dashboard, (0, 0), (dashboard.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(
        dashboard, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.62, (80, 255, 80), 2, cv2.LINE_AA,
    )
    return cv2.resize(dashboard, (output_width, output_height))


def stream_dataset(
    model, dataset, indices, input_mode="fusion", device="auto",
    score_threshold=0.3, top_k=50, min_center_distance=1.5,
    output_video=None, video_fps=10.0, output_size=(1200, 360),
    display_every=1, frame_callback=None, draw_heading=True,
):
    """Run streaming inference, optionally save MP4, and return timing records."""
    if not isinstance(display_every, int) or display_every <= 0:
        raise ValueError("display_every must be a positive integer")
    if not np.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("video_fps must be finite and positive")
    resolved_device = resolve_inference_device(device)
    indices = list(indices)
    if not indices:
        raise ValueError("indices must contain at least one frame")

    timing_records, frame_records = [], []
    writer = None
    try:
        for position, sample_index in enumerate(indices):
            pipeline_start = time.perf_counter()
            sample = dataset[sample_index]

            start = time.perf_counter()
            bev = build_model_input(sample, input_mode)
            inputs = torch.from_numpy(bev).unsqueeze(0).to(resolved_device)
            if resolved_device.type == "cuda":
                torch.cuda.synchronize()
            preprocessing = time.perf_counter() - start

            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(inputs)
            if resolved_device.type == "cuda":
                torch.cuda.synchronize()
            network = time.perf_counter() - start

            start = time.perf_counter()
            detections = decode_predictions(
                outputs, score_threshold, top_k, min_center_distance
            )
            decoding = time.perf_counter() - start
            partial_pipeline = time.perf_counter() - pipeline_start

            start = time.perf_counter()
            dashboard = create_stream_dashboard(
                sample, bev, detections,
                {"network": network, "pipeline": partial_pipeline},
                output_size, draw_heading,
            )
            rendering = time.perf_counter() - start
            pipeline = time.perf_counter() - pipeline_start
            timings = {
                "preprocessing": preprocessing, "network": network,
                "decoding": decoding, "rendering": rendering, "pipeline": pipeline,
            }
            timing_records.append(timings)
            frame_record = {
                "position": position, "index": sample_index, "id": sample["id"],
                "detections": detections, "timings": timings,
            }
            frame_records.append(frame_record)

            if output_video is not None:
                if writer is None:
                    path = Path(output_video)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps,
                        (dashboard.shape[1], dashboard.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"could not create output video: {path}")
                writer.write(dashboard)

            if frame_callback is not None and position % display_every == 0:
                frame_callback(dashboard, frame_record, len(indices))
    finally:
        if writer is not None:
            writer.release()
    return {"frames": frame_records, "timings": timing_records}


def summarize_stream_timings(stream_result):
    """Return mean, median, and effective FPS for each measured stage."""
    records = stream_result.get("timings", [])
    if not records:
        raise ValueError("stream_result contains no timing records")
    summary = {}
    for key in ("preprocessing", "network", "decoding", "rendering", "pipeline"):
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        mean = float(values.mean())
        summary[key] = {
            "mean_seconds": mean,
            "median_seconds": float(np.median(values)),
            "fps": 1.0 / mean,
        }
    return summary
