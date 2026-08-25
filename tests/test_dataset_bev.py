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


class BevProjectionTests(unittest.TestCase):
    def test_shape_dtype_channels_and_boundaries(self):
        points = np.array([
            [1.01, -39.99, -2.0, 0.25],
            [1.02, -39.98, 0.0, 0.75],
            [70.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, np.nan, 1.0],
        ], dtype=np.float32)
        bev = bev_projection(points)

        self.assertEqual(bev.shape, (3, *BEV_SHAPE))
        self.assertEqual(bev.dtype, np.float32)
        self.assertTrue(np.isfinite(bev).all())
        self.assertAlmostEqual(float(bev[0, 5, 0]), 0.75)
        self.assertAlmostEqual(float(bev[2, 5, 0]), 0.75)
        self.assertAlmostEqual(float(bev[1, 5, 0]), np.log(3) / np.log(65))

    def test_empty_and_invalid_inputs(self):
        empty = bev_projection(np.empty((0, 4), dtype=np.float32))
        self.assertEqual(empty.shape, (3, *BEV_SHAPE))
        self.assertFalse(empty.any())
        with self.assertRaises(ValueError):
            bev_projection(np.empty((4, 3), dtype=np.float32))


class DatasetTests(unittest.TestCase):
    def make_split(self, root, split, with_labels):
        split_dir = Path(root) / split
        (split_dir / "velodyne").mkdir(parents=True)
        (split_dir / "calib").mkdir()
        points = np.array([[1.0, 2.0, 0.0, 0.5]], dtype=np.float32)
        points.tofile(split_dir / "velodyne" / "000000.bin")
        (split_dir / "calib" / "000000.txt").write_text(CALIBRATION)
        if with_labels:
            (split_dir / "label_2").mkdir()
            label = "Car 0.0 0 0.1 1 2 3 4 1.5 1.6 3.8 1 2 20 0.5\n"
            (split_dir / "label_2" / "000000.txt").write_text(label)

    def test_training_and_testing_samples(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_split(root, "training", with_labels=True)
            self.make_split(root, "testing", with_labels=False)

            training = KittiDataset(root, "training")[0]
            testing = KittiDataset(root, "testing")[0]

            self.assertEqual(training["points"].shape, (1, 4))
            self.assertEqual(len(training["labels"]), 1)
            self.assertEqual(training["labels"][0]["occluded"], 0)
            self.assertEqual(testing["labels"], [])
            self.assertEqual(testing["calib"]["p2"].shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
