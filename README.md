# KITTI LiDAR Detection

A compact educational baseline for detecting cars in KITTI LiDAR point clouds using a bird's-eye-view (BEV) representation and a CenterNet-style PyTorch detector.

## Features

- Load KITTI Velodyne points, calibration records, and optional labels.
- Convert KITTI camera boxes into LiDAR and BEV coordinates.
- Rasterize height, density, and intensity into a BEV tensor.
- Generate center heatmaps plus offset, size, and rotation targets for cars.
- Train a multi-scale encoder-decoder with augmentation and validation-based checkpoints.
- Decode local heatmap peaks into metric-space BEV detections.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,test]"
```

On Windows PowerShell, activate the environment with `\.venv\Scripts\Activate.ps1`.

## Dataset layout

Download the object-detection data from the [KITTI website](https://www.cvlibs.net/datasets/kitti/eval_object.php) and extract it as follows:

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

Testing data normally has no labels. In that case, `KittiDataset(..., split="testing")` returns an empty `labels` list.

## BEV pipeline

```python
from pathlib import Path

from src.dataset.kitti_dataset import KittiDataset

dataset = KittiDataset(Path("data/KITTI"), split="training")
sample = dataset[0]
bev, targets = dataset.get_training_sample(0)

print(sample["points"].shape)
print(bev.shape)                 # (3, 350, 400)
print(targets["heatmap"].shape) # (350, 400)
```

The default grid covers x = 070 m, y = -4040 m, and z = -31 m with 0.2 m voxels. BEV channels are ordered height, density, intensity.

## Training

The training module is safe to import and can be run from the repository root:

```bash
python -m src.training.training_script \
  --data-root data/KITTI \
  --epochs 30 \
  --batch-size 6 \
  --subset-size 1000 \
  --validation-fraction 0.2 \
  --flip-probability 0.5 \
  --learning-rate 0.001 \
  --device auto \
  --seed 0 \
  --output outputs/bev_detector_multiscale.pth
```

Subset selection is seeded and randomized; use --subset-size 0 for the full dataset. Training reports validation loss plus center precision/recall and restores the best validation epoch.

The checkpoint contains `model_state_dict`, the effective training configuration, and per-epoch loss history. The model uses two downsampling stages, dilated context, skip-connected decoding, and a low foreground heatmap prior. It remains an educational baseline rather than a benchmark-quality KITTI detector.

## Decoding

```python
from src.inference.decode import decode_predictions

model.eval()
outputs = model(bev_tensor)
detections = decode_predictions(outputs, score_threshold=0.5, top_k=50)
```

Decoding currently supports one sample at a time. It keeps local heatmap maxima, merges nearby single-class centers, filters invalid sizes, and returns score, center, length, width, and yaw for each detection.

## Notebooks

The ten numbered notebooks progress from raw KITTI inspection through BEV training and decoding. Start Jupyter from the repository root so imports and data paths resolve consistently:

```bash
jupyter notebook
```

## Tests

The automated tests use temporary KITTI-style fixtures and do not require the full dataset:

```bash
python -m unittest discover -s tests -v
```

The suite covers dataset validation, BEV construction, geometry, targets, model/loss behavior, decoding, and checkpoint utilities.
