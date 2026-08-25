# KITTI LiDAR Detection

Utilities and notebooks for exploring the KITTI object-detection benchmark with LiDAR point clouds. The project currently provides dataset loading, camera/LiDAR calibration transforms, 3D box conversion, and bird's-eye-view (BEV) preprocessing.

## Features

- Load KITTI Velodyne point clouds from `.bin` files.
- Parse KITTI calibration files and object labels.
- Convert rectified camera coordinates to LiDAR coordinates.
- Rasterize point clouds into three BEV channels: height, density, and intensity.
- Build BEV targets for `Car` objects inside the configured detection region.
- Explore the pipeline through numbered Jupyter notebooks.

## Project Layout

```text
src/
  config/         Point-cloud range and voxel-size constants
  dataset/        KITTI dataset access
  geometry/       Coordinate transforms and 3D box helpers
  preprocessing/  BEV projection
  targets/        BEV ground-truth target generation
notebooks/        Step-by-step LiDAR and BEV exploration
tests/            Unit tests for dataset loading and BEV projection
data/KITTI/       Local KITTI data (not included in this guide)
outputs/          Generated results
```

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib and Jupyter for the notebooks

Create a virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib jupyter
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## KITTI Dataset

Download the KITTI object-detection data from the [official KITTI website](https://www.cvlibs.net/datasets/kitti/eval_object.php) and place or extract it under `data/KITTI/` with this structure:

```text
data/KITTI/
  training/
    velodyne/*.bin
    calib/*.txt
    label_2/*.txt
  testing/
    velodyne/*.bin
    calib/*.txt
```

Training labels are optional for dataset access, while testing data normally has no `label_2` directory. Sample IDs are matched by filename, for example `000000.bin` with `000000.txt`.

## Usage

Run this from the repository root:

```python
from pathlib import Path

from src.dataset.kitti_dataset import KittiDataset
from src.preprocessing.bev import bev_projection

dataset = KittiDataset(Path("data/KITTI"), split="training")
sample = dataset[0]
bev = bev_projection(sample["points"])

print(len(dataset))
print(sample["id"], sample["points"].shape)
print(bev.shape)  # (3, 350, 400)
```

The default BEV configuration covers:

| Axis | Range | Voxel size |
| --- | --- | --- |
| `x` | `0.0` to `70.0` m | `0.2` m |
| `y` | `-40.0` to `40.0` m | `0.2` m |
| `z` | `-3.0` to `1.0` m | `0.2` m |

The resulting tensor has channels in the order `height`, `density`, `intensity`.

## Notebooks

Open Jupyter from the repository root:

```bash
jupyter notebook
```

The notebooks build from raw point-cloud inspection through labels, calibration, image projection, 3D boxes, dataset access, and BEV targets:

1. `01_kitti_lidar_basics.ipynb`
2. `02_kitti_lidar_label.ipynb`
3. `03_kitti_lidar_calib.ipynb`
4. `04_kitti_lidar_2d_proj.ipynb`
5. `05_kitti_3d_boxes.ipynb`
6. `06_kitti_lidar_dataset.ipynb`
7. `07_bev_ground_truth.ipynb`
8. `08_bev_testing.ipynb`

## Tests

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests use temporary KITTI-style files, so they do not require the full dataset.

## Current Scope

This repository is a preprocessing and study baseline. It does not currently include a neural-network training loop, evaluation pipeline, or command-line inference entry point.