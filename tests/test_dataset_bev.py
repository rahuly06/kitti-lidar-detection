import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config.constants import BEV_SHAPE
from src.dataset.kitti_dataset import KittiDataset
from src.preprocessing.bev import bev_projection


CALIBRATION = """P0: 0 0 0 0 0 0 0 0 0 0 0 0
P1: 0 0 0 0 0 0 0 0 0 0 0 0
P2: 1 0 0 0 0 1 0 0 0 0 1 0
P3: 0 0 0 0 0 0 0 0 0 0 0 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0
Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0
"""
LABEL = "Car 0.0 0 0.1 1 2 3 4 1.5 1.6 3.8 10 0 20 0.5\n"


def make_split(root, split, labels=True, points=None):
    split_dir = Path(root) / split
    (split_dir / "velodyne").mkdir(parents=True)
    (split_dir / "calib").mkdir()
    if points is None:
        points = np.array([[1.0, 2.0, 0.0, 0.5]], dtype=np.float32)
    points.tofile(split_dir / "velodyne" / "000000.bin")
    (split_dir / "calib" / "000000.txt").write_text(CALIBRATION)
    if labels:
        (split_dir / "label_2").mkdir()
        (split_dir / "label_2" / "000000.txt").write_text(LABEL)


class BevProjectionTests(unittest.TestCase):
    def test_shape_dtype_channels_and_boundaries(self):
        points = np.array(
            [
                [1.01, -39.99, -2.0, 0.25],
                [1.02, -39.98, 0.0, 0.75],
                [70.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, np.nan, 1.0],
            ],
            dtype=np.float32,
        )
        bev = bev_projection(points)
        self.assertEqual(bev.shape, (3, *BEV_SHAPE))
        self.assertEqual(bev.dtype, np.float32)
        self.assertTrue(np.isfinite(bev).all())
        self.assertAlmostEqual(float(bev[0, 5, 0]), 0.5)
        self.assertAlmostEqual(float(bev[2, 5, 0]), 0.75)
        self.assertAlmostEqual(float(bev[1, 5, 0]), np.log(3) / np.log(65))

    def test_empty_and_invalid_inputs(self):
        empty = bev_projection(np.empty((0, 4), dtype=np.float32))
        self.assertEqual(empty.shape, (3, *BEV_SHAPE))
        self.assertFalse(empty.any())
        with self.assertRaises(ValueError):
            bev_projection(np.empty((4, 3), dtype=np.float32))
        with self.assertRaises(TypeError):
            bev_projection(np.array([["x", "y", "z", "i"]]))


class DatasetTests(unittest.TestCase):
    def test_training_and_testing_samples(self):
        with tempfile.TemporaryDirectory() as root:
            make_split(root, "training", labels=True)
            make_split(root, "testing", labels=False)
            dataset = KittiDataset(root, "training")
            training = dataset[-1]
            testing = KittiDataset(root, "testing")[0]
            self.assertEqual(training["points"].shape, (1, 4))
            self.assertEqual(training["labels"][0]["occluded"], 0)
            self.assertEqual(testing["labels"], [])
            self.assertEqual(testing["calib"]["p2"].shape, (3, 4))
            bev, targets = dataset.get_training_sample(0)
            self.assertEqual(bev.shape, (3, *BEV_SHAPE))
            self.assertEqual(targets["heatmap"].shape, BEV_SHAPE)

    def test_invalid_split_directories_and_empty_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                KittiDataset(root, "validation")
            with self.assertRaises(FileNotFoundError):
                KittiDataset(root, "training")
            (Path(root) / "training" / "velodyne").mkdir(parents=True)
            (Path(root) / "training" / "calib").mkdir()
            dataset = KittiDataset(root)
            self.assertEqual(len(dataset), 0)
            with self.assertRaises(IndexError):
                dataset[0]

    def test_malformed_records(self):
        with tempfile.TemporaryDirectory() as root:
            make_split(root, "training")
            dataset = KittiDataset(root)
            np.arange(3, dtype=np.float32).tofile(dataset.velodyne_dir / "000000.bin")
            with self.assertRaises(ValueError):
                dataset.load_velodyne("000000")
            (dataset.calib_dir / "000000.txt").write_text("P2: 1 2\n")
            with self.assertRaises(ValueError):
                dataset.load_calibration("000000")
            (dataset.label_dir / "000000.txt").write_text("Car too short\n")
            with self.assertRaises(ValueError):
                dataset.load_labels("000000")

    def test_training_targets_require_labels(self):
        with tempfile.TemporaryDirectory() as root:
            make_split(root, "testing", labels=False)
            with self.assertRaises(ValueError):
                KittiDataset(root, "testing").get_training_sample(0)


if __name__ == "__main__":
    unittest.main()
