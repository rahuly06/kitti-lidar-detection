import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from src.models.bev_detector import BEVDetector
from src.training.training_script import (
    DetectionView,
    build_parser,
    save_checkpoint,
    seed_everything,
    split_indices,
    train,
)


class TinyDataset:
    has_labels = True

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "points": np.array([[1.0, 2.0, 0.0, 0.5]], dtype=np.float32),
            "labels": [],
            "calib": {
                "tr_velo_to_cam": np.column_stack((np.eye(3), np.zeros(3))),
                "r0_rect": np.eye(3),
            },
        }


class FlipDataset:
    has_labels = True

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "points": np.array([[10.0, 2.0, 0.0, 0.5]], dtype=np.float32),
            "labels": [
                {
                    "type": "Car",
                    "dimensions": np.array([1.5, 1.6, 3.8]),
                    "location": np.array([-2.0, 0.0, 10.0]),
                    "rotation_y": 0.0,
                }
            ],
            "calib": {
                "tr_velo_to_cam": np.array(
                    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=np.float32
                ),
                "r0_rect": np.eye(3),
            },
        }


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 7, kernel_size=1)

    def forward(self, inputs):
        output = self.head(inputs)
        return {
            "heatmap": output[:, :1],
            "offset": output[:, 1:3],
            "size": output[:, 3:5],
            "rotation": output[:, 5:7],
        }


class TrainingUtilityTests(unittest.TestCase):
    def test_parser_seed_split_and_checkpoint(self):
        args = build_parser().parse_args(["--epochs", "1", "--subset-size", "1", "--device", "cpu"])
        self.assertEqual(args.epochs, 1)
        seed_everything(7)
        first = torch.rand(1)
        seed_everything(7)
        torch.testing.assert_close(first, torch.rand(1))
        train_a, validation_a = split_indices(20, 10, 0.2, 4)
        train_b, validation_b = split_indices(20, 10, 0.2, 4)
        self.assertEqual((train_a, validation_a), (train_b, validation_b))
        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(validation_a), 2)
        self.assertTrue(set(train_a).isdisjoint(validation_a))

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "nested" / "model.pth"
            save_checkpoint(BEVDetector(), path, {"epochs": 1}, [{"total": 2.0}], {"train_indices": [0]})
            checkpoint = torch.load(path, weights_only=False)
            self.assertIn("model_state_dict", checkpoint)
            self.assertEqual(checkpoint["config"]["epochs"], 1)
            self.assertEqual(checkpoint["split"]["train_indices"], [0])

    def test_forced_lateral_flip(self):
        plain_bev, plain_targets = DetectionView(FlipDataset(), [0], augment=False)[0]
        flip_bev, flip_targets = DetectionView(
            FlipDataset(), [0], augment=True, flip_probability=1.0
        )[0]
        plain_point = np.argwhere(plain_bev[2] > 0)[0]
        flip_point = np.argwhere(flip_bev[2] > 0)[0]
        self.assertEqual(int(plain_point[0]), int(flip_point[0]))
        self.assertEqual(int(plain_point[1] + flip_point[1]), plain_bev.shape[2])
        plain_center = np.argwhere(plain_targets["regression_mask"])[0]
        flip_center = np.argwhere(flip_targets["regression_mask"])[0]
        self.assertEqual(int(plain_center[0]), int(flip_center[0]))
        self.assertEqual(int(plain_center[1] + flip_center[1]), plain_targets["heatmap"].shape[1])

    def test_tiny_training_smoke(self):
        with patch("src.training.training_script.BEVDetector", TinyDetector):
            model, history, device, split = train(
                TinyDataset(),
                epochs=1,
                batch_size=1,
                subset_size=1,
                learning_rate=1e-3,
                device="cpu",
                seed=3,
                validation_fraction=0,
                flip_probability=0,
            )
        self.assertIsInstance(model, TinyDetector)
        self.assertEqual(str(device), "cpu")
        self.assertEqual(len(history), 1)
        self.assertEqual(split["train_indices"], [0])
        self.assertTrue(np.isfinite(history[0]["train"]["total"]))


if __name__ == "__main__":
    unittest.main()
