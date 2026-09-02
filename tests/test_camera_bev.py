import unittest

import numpy as np

from src.config.constants import BEV_SHAPE
from src.preprocessing.camera_bev import aggregate_rgb_points_to_bev


class CameraBevTests(unittest.TestCase):
    def test_mean_rgb_visibility_shape_and_dtype(self):
        points = np.array(
            [
                [1.01, -39.99, 0.0, 0.2, 0.2, 0.4, 0.6],
                [1.02, -39.98, 0.1, 0.8, 0.6, 0.8, 1.0],
                [69.99, 39.99, 0.0, 0.5, 1.0, 0.0, 0.5],
            ],
            dtype=np.float32,
        )
        camera_bev = aggregate_rgb_points_to_bev(points)
        self.assertEqual(camera_bev.shape, (4, *BEV_SHAPE))
        self.assertEqual(camera_bev.dtype, np.float32)
        np.testing.assert_allclose(camera_bev[:3, 5, 0], [0.4, 0.6, 0.8])
        self.assertEqual(float(camera_bev[3, 5, 0]), 1.0)
        np.testing.assert_allclose(camera_bev[:3, 349, 399], [1.0, 0.0, 0.5])
        self.assertEqual(float(camera_bev[3, 349, 399]), 1.0)

    def test_empty_outside_and_nonfinite_points_are_ignored(self):
        points = np.array(
            [
                [70.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 40.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [np.nan, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        camera_bev = aggregate_rgb_points_to_bev(points)
        self.assertFalse(camera_bev.any())
        empty = aggregate_rgb_points_to_bev(np.empty((0, 7), dtype=np.float32))
        self.assertEqual(empty.shape, (4, *BEV_SHAPE))
        self.assertFalse(empty.any())

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_rgb_points_to_bev(np.empty((2, 6), dtype=np.float32))
        with self.assertRaises(TypeError):
            aggregate_rgb_points_to_bev(np.full((2, 7), "bad"))
        invalid_rgb = np.zeros((1, 7), dtype=np.float32)
        invalid_rgb[0, 0] = 1.0
        invalid_rgb[0, 4] = 1.1
        with self.assertRaises(ValueError):
            aggregate_rgb_points_to_bev(invalid_rgb)


if __name__ == "__main__":
    unittest.main()
