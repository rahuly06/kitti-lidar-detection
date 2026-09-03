"""Training pipeline for the multi-scale BEV detector."""

import argparse
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config.constants import POINT_CLOUD_RANGE, VOXEL_SIZE
from src.dataset.kitti_dataset import KittiDataset
from src.geometry.boxes import get_lidar_boxes
from src.inference.decode import decode_predictions
from src.losses.detection_loss import DetectionLoss
from src.models.bev_detector import BEVDetector
from src.preprocessing.bev import bev_projection
from src.preprocessing.fusion import build_fused_bev
from src.targets.bev_targets import create_bev_targets


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "KITTI"
TARGET_KEYS = ("heatmap", "offset", "size", "rotation", "regression_mask")


def seed_everything(seed):
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


class DetectionView(Dataset):
    """Create BEV tensors and targets for a fixed set of KITTI indices."""

    def __init__(
        self,
        dataset,
        indices,
        augment=False,
        flip_probability=0.5,
        input_mode="lidar",
    ):
        if not 0 <= flip_probability <= 1:
            raise ValueError("flip_probability must lie in [0, 1]")
        if input_mode not in {"lidar", "fusion"}:
            raise ValueError("input_mode must be 'lidar' or 'fusion'")
        if input_mode == "fusion" and not dataset.load_images:
            raise ValueError("fusion input requires a dataset with load_images=True")
        self.dataset = dataset
        self.indices = list(indices)
        self.augment = augment
        self.flip_probability = flip_probability
        self.input_mode = input_mode

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        sample = self.dataset[self.indices[position]]
        points = sample["points"]
        boxes = get_lidar_boxes(sample["labels"], sample["calib"])

        flip = self.augment and torch.rand(()).item() < self.flip_probability
        if self.input_mode == "fusion":
            bev = build_fused_bev(points, sample["image"], sample["calib"])
            if flip:
                # The second spatial dimension is lateral. Flip every channel
                # together so LiDAR, RGB, and visibility remain aligned.
                bev = np.flip(bev, axis=2).copy()
        else:
            if flip:
                points = points.copy()
                points[:, 1] *= -1
            bev = bev_projection(points)

        if flip:
            boxes = [
                {
                    **box,
                    "center_y": -box["center_y"],
                    "yaw": -box["yaw"],
                }
                for box in boxes
            ]

        return bev, create_bev_targets(boxes)


def collate_detection_batch(items):
    if not items:
        raise ValueError("cannot collate an empty batch")
    bevs, targets = zip(*items)
    inputs = torch.from_numpy(np.stack(bevs)).float()
    batch_targets = {}
    for key in TARGET_KEYS:
        values = np.stack([target[key] for target in targets])
        tensor = torch.from_numpy(values).float()
        if key in {"heatmap", "regression_mask"}:
            tensor = tensor.unsqueeze(1)
        batch_targets[key] = tensor
    return inputs, batch_targets


def create_batch(dataset, indices, device, input_mode="lidar"):
    """Backward-compatible direct batch construction."""
    view = DetectionView(
        dataset,
        indices,
        augment=False,
        input_mode=input_mode,
    )
    inputs, targets = collate_detection_batch([view[index] for index in range(len(view))])
    return inputs.to(device), {key: value.to(device) for key, value in targets.items()}


def split_indices(dataset_size, subset_size, validation_fraction, seed):
    if dataset_size <= 0:
        raise ValueError("training dataset is empty")
    if subset_size < 0:
        raise ValueError("subset_size must be zero (all data) or positive")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must lie in [0, 1)")

    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(dataset_size, generator=generator).tolist()
    selected = shuffled if subset_size == 0 else shuffled[: min(subset_size, dataset_size)]
    if validation_fraction == 0 or len(selected) == 1:
        return selected, []
    validation_count = max(1, round(len(selected) * validation_fraction))
    validation_count = min(validation_count, len(selected) - 1)
    return selected[validation_count:], selected[:validation_count]


def _move_targets(targets, device):
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


def _accumulate_center_metrics(outputs, targets, threshold=0.3, match_distance=2.0):
    true_positives = false_positives = false_negatives = 0
    batch_size = outputs["heatmap"].shape[0]
    x_min, _ = POINT_CLOUD_RANGE["x"]
    y_min, _ = POINT_CLOUD_RANGE["y"]

    for batch_index in range(batch_size):
        sample_outputs = {key: value[batch_index : batch_index + 1] for key, value in outputs.items()}
        detections = decode_predictions(sample_outputs, threshold, top_k=100)
        mask = targets["regression_mask"][batch_index, 0] > 0
        cells = torch.nonzero(mask, as_tuple=False)
        ground_truth = []
        for ix, iy in cells:
            dx = float(targets["offset"][batch_index, 0, ix, iy])
            dy = float(targets["offset"][batch_index, 1, ix, iy])
            ground_truth.append(
                (
                    (int(ix) + dx) * VOXEL_SIZE["x"] + x_min,
                    (int(iy) + dy) * VOXEL_SIZE["y"] + y_min,
                )
            )

        unmatched = set(range(len(ground_truth)))
        for detection in detections:
            candidates = [
                (
                    math.hypot(
                        detection["center_x"] - ground_truth[index][0],
                        detection["center_y"] - ground_truth[index][1],
                    ),
                    index,
                )
                for index in unmatched
            ]
            if candidates:
                distance, matched_index = min(candidates)
            else:
                distance, matched_index = math.inf, None
            if distance <= match_distance:
                true_positives += 1
                unmatched.remove(matched_index)
            else:
                false_positives += 1
        false_negatives += len(unmatched)
    return true_positives, false_positives, false_negatives


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("total", "heatmap", "offset", "size", "rotation")}
    batches = true_positives = false_positives = false_negatives = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = _move_targets(targets, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].item())
            if not training:
                tp, fp, fn = _accumulate_center_metrics(outputs, targets)
                true_positives += tp
                false_positives += fp
                false_negatives += fn
            batches += 1

    metrics = {key: value / batches for key, value in totals.items()}
    if not training:
        metrics["precision"] = true_positives / max(1, true_positives + false_positives)
        metrics["recall"] = true_positives / max(1, true_positives + false_negatives)
    return metrics


def train(
    dataset,
    epochs=30,
    batch_size=2,
    learning_rate=1e-3,
    subset_size=0,
    device="auto",
    seed=0,
    validation_fraction=0.2,
    flip_probability=0.5,
    num_workers=0,
    input_mode="lidar",
):
    if not dataset.has_labels:
        raise ValueError("training requires a dataset with labels")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or num_workers < 0:
        raise ValueError("epochs, batch_size, learning_rate must be positive and num_workers non-negative")
    if input_mode not in {"lidar", "fusion"}:
        raise ValueError("input_mode must be 'lidar' or 'fusion'")
    if input_mode == "fusion" and not dataset.load_images:
        raise ValueError("fusion training requires load_images=True")

    seed_everything(seed)
    device = resolve_device(device)
    train_indices, validation_indices = split_indices(
        len(dataset), subset_size, validation_fraction, seed
    )
    train_view = DetectionView(
        dataset,
        train_indices,
        augment=True,
        flip_probability=flip_probability,
        input_mode=input_mode,
    )
    validation_view = DetectionView(
        dataset,
        validation_indices,
        augment=False,
        input_mode=input_mode,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_detection_batch,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_view, shuffle=True, generator=generator, **loader_options)
    validation_loader = (
        DataLoader(validation_view, shuffle=False, **loader_options) if validation_indices else None
    )

    input_channels = 7 if input_mode == "fusion" else 3
    model = BEVDetector(input_channels=input_channels).to(device)
    criterion = DetectionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    best_score = math.inf
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = (
            run_epoch(model, validation_loader, criterion, device)
            if validation_loader is not None
            else {}
        )
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        selection_score = validation_metrics.get("total", train_metrics["total"])
        if selection_score < best_score:
            best_score = selection_score
            best_state = copy.deepcopy(model.state_dict())

        message = f"Epoch {epoch + 1:03d}/{epochs} | train {train_metrics['total']:.4f}"
        if validation_metrics:
            message += (
                f" | val {validation_metrics['total']:.4f}"
                f" | precision {validation_metrics['precision']:.3f}"
                f" | recall {validation_metrics['recall']:.3f}"
            )
        print(message, flush=True)

    model.load_state_dict(best_state)
    split = {"train_indices": train_indices, "validation_indices": validation_indices}
    return model, history, device, split


def save_checkpoint(model, output_path, config, history, split=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
            "split": split or {},
        },
        output_path,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--subset-size", type=int, default=0, help="0 uses all samples")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-mode",
        choices=("lidar", "fusion"),
        default="lidar",
        help="use 3-channel LiDAR BEV or 7-channel LiDAR-camera BEV",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/bev_detector_multiscale.pth"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_channels = 7 if args.input_mode == "fusion" else 3
    dataset = KittiDataset(
        args.data_root,
        split="training",
        load_images=args.input_mode == "fusion",
    )
    config = {
        "data_root": str(args.data_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "subset_size": args.subset_size,
        "validation_fraction": args.validation_fraction,
        "flip_probability": args.flip_probability,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "device": args.device,
        "seed": args.seed,
        "input_mode": args.input_mode,
        "input_channels": input_channels,
        "architecture": "multiscale_encoder_decoder_v1",
    }
    model, history, actual_device, split = train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subset_size=args.subset_size,
        device=args.device,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        flip_probability=args.flip_probability,
        num_workers=args.num_workers,
        input_mode=args.input_mode,
    )
    config["resolved_device"] = str(actual_device)
    config["train_samples"] = len(split["train_indices"])
    config["validation_samples"] = len(split["validation_indices"])
    save_checkpoint(model, args.output, config, history, split)
    print(f"Best model saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
