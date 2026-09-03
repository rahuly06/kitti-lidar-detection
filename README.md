# KITTI LiDAR–Camera Car Detection

A lightweight project for detecting cars from KITTI LiDAR point clouds. The detector works in bird's-eye view (BEV) and supports two input modes:

- **LiDAR only:** height, point density, and reflectance intensity.
- **LiDAR–camera fusion:** the LiDAR features plus RGB information projected from the left color camera.

The project covers the complete workflow from loading and aligning sensor data to training, evaluation, visualization, and streaming inference.

## How it works

```text
LiDAR point cloud ────────────────→ LiDAR BEV ───────┐
                                                     ├→ BEV detector → car boxes
Camera image → project RGB onto LiDAR → camera BEV ──┘
```

LiDAR supplies the distance and geometry. Camera pixels are associated with LiDAR points using KITTI calibration, so their colors can be placed into the same BEV grid. A small encoder–decoder CNN then predicts car centers, dimensions, orientation, and confidence.

The detector is single-class and recognizes only the KITTI `Car` category.

## Project highlights

- KITTI Object Detection training and testing loaders.
- KITTI Raw synchronized-sequence loader.
- LiDAR-to-camera projection using KITTI calibration.
- Three-channel LiDAR BEV and seven-channel fusion BEV.
- Lightweight U-Net-like, center-based detector.
- Training and validation with best-checkpoint saving.
- LiDAR-versus-fusion evaluation and confidence-threshold sweep.
- Conventional forward-up BEV visualization.
- Streaming inference with optional MP4 output and latency reporting.

On the 293 validation frames that were unseen by both trained models, fusion improved the center-based diagnostic F1 score from `0.815` to `0.885` at a confidence threshold of `0.30`. These are project validation metrics, not official KITTI Average Precision.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,test]"
```

## KITTI Object Detection data

Download the left color images, Velodyne point clouds, calibration files, and training labels from the [KITTI Object Detection website](https://www.cvlibs.net/datasets/kitti/eval_object.php).

```text
data/KITTI/
├── training/
│   ├── image_2/*.png
│   ├── velodyne/*.bin
│   ├── calib/*.txt
│   └── label_2/*.txt
└── testing/
    ├── image_2/*.png
    ├── velodyne/*.bin
    └── calib/*.txt
```

The official testing split has no public ground-truth labels. It can be used for inference and visual inspection, but local accuracy metrics must be calculated on held-out training frames.

## Training

LiDAR-only training:

```bash
python -m src.training.training_script \
  --data-root data/KITTI \
  --input-mode lidar \
  --epochs 30 \
  --batch-size 4 \
  --subset-size 0 \
  --validation-fraction 0.2 \
  --flip-probability 0.5 \
  --learning-rate 0.001 \
  --device cuda \
  --seed 0 \
  --output outputs/bev_detector_multiscale_full.pth
```

Fusion training:

```bash
python -m src.training.training_script \
  --data-root data/KITTI \
  --input-mode fusion \
  --epochs 30 \
  --batch-size 4 \
  --subset-size 0 \
  --validation-fraction 0.2 \
  --flip-probability 0.5 \
  --learning-rate 0.001 \
  --device cuda \
  --seed 42 \
  --output outputs/bev_detector_fusion_full_seed42.pth
```

Use a smaller batch size if GPU memory is limited. Training saves the best validation checkpoint rather than simply keeping the final epoch.

## Inference and evaluation

Reusable inference utilities are in `src/inference/pipeline.py`. They load either checkpoint, build the correct BEV input, decode detections, and calculate lightweight validation metrics.

The main evaluation notebooks are:

- `17_kitti_inference_fusion.ipynb` — inspect fusion predictions.
- `18_lidar_vs_fusion_comparison.ipynb` — compare both models on shared validation frames and test confidence thresholds.
- `19_kitti_fusion_stream.ipynb` — replay a synchronized KITTI Raw drive.

In notebook 19, switch models with:

```python
INPUT_MODE = "fusion"  # or "lidar"
```

The streaming module in `src/inference/streaming.py` handles conventional forward-up BEV rendering, heading arrows, MP4 writing, and timing measurements.

## KITTI Raw streaming data

Notebook 19 uses the synchronized and rectified Raw drive:

```text
data/KITTI/2011_09_26/
├── calib_cam_to_cam.txt
├── calib_velo_to_cam.txt
└── 2011_09_26_drive_0011_sync/
    ├── image_02/data/*.png
    └── velodyne_points/data/*.bin
```

The drive contains 233 synchronized frames, or approximately 23.3 seconds at 10 Hz.

## Tests

```bash
python -m pytest -q
```

Tests cover dataset loading, geometry, BEV construction, fusion, targets, the detector, losses, decoding, training utilities, and inference helpers.

## Scope

This is a dependable learning and research baseline, not a production autonomous-driving system. It demonstrates how calibrated LiDAR and camera data can be combined for BEV car detection while keeping the implementation small enough to understand from end to end.
