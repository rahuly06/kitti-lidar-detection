import unittest

import numpy as np
import torch

from src.config.constants import BEV_SHAPE
from src.models.bev_detector import BEVDetector
from src.preprocessing.fusion import (
    FUSED_BEV_CHANNELS,
    build_fused_bev,
    paint_lidar_points_with_rgb,
)


CALIBRATION = {
    "tr_velo_to_cam": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype=np.float32,
    ),
    "r0_rect": np.eye(3, dtype=np.float32),
    "p2": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype=np.float32,
    ),
}


class FusionTests(unittest.TestCase):
    def test_painting_preserves_point_pixel_correspondence(self):
        points = np.array(
            [
                [2.0, 2.0, 1.0, 0.5],
                [4.0, 2.0, 2.0, 0.8],
                [1.0, 1.0, -1.0, 0.2],
            ],
            dtype=np.float32,
        )
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[2, 2] = [255, 128, 0]
        image[1, 2] = [0, 64, 255]

        painted, pixels, depths = paint_lidar_points_with_rgb(
            points,
            image,
            CALIBRATION,
        )

        self.assertEqual(painted.shape, (2, 7))
        np.testing.assert_allclose(pixels, [[2.0, 2.0], [2.0, 1.0]])
        np.testing.assert_allclose(depths, [1.0, 2.0])
        np.testing.assert_allclose(painted[0, 4:7], [1.0, 128 / 255, 0.0])
        np.testing.assert_allclose(painted[1, 4:7], [0.0, 64 / 255, 1.0])

    def test_build_fused_bev_shape_channels_and_visibility(self):
        points = np.array(
            [[2.0, 2.0, 1.0, 0.5]],
            dtype=np.float32,
        )
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[2, 2] = [255, 128, 0]

        fused_bev = build_fused_bev(points, image, CALIBRATION)

        self.assertEqual(len(FUSED_BEV_CHANNELS), 7)
        self.assertEqual(fused_bev.shape, (7, *BEV_SHAPE))
        self.assertEqual(fused_bev.dtype, np.float32)
        self.assertTrue(np.isfinite(fused_bev).all())
        np.testing.assert_allclose(
            fused_bev[3:6, 10, 210],
            [1.0, 128 / 255, 0.0],
        )
        self.assertEqual(float(fused_bev[6, 10, 210]), 1.0)

    def test_invalid_inputs_are_rejected(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            paint_lidar_points_with_rgb(
                np.empty((1, 3), dtype=np.float32),
                image,
                CALIBRATION,
            )
        with self.assertRaises(ValueError):
            paint_lidar_points_with_rgb(
                np.empty((0, 4), dtype=np.float32),
                np.empty((4, 4), dtype=np.uint8),
                CALIBRATION,
            )
        with self.assertRaises(TypeError):
            paint_lidar_points_with_rgb(
                np.empty((0, 4), dtype=np.float32),
                np.full((4, 4, 3), "bad"),
                CALIBRATION,
            )

    def test_detector_accepts_configurable_fusion_channels(self):
        model = BEVDetector(input_channels=7)
        inputs = torch.zeros((1, 7, 32, 40), dtype=torch.float32)

        outputs = model(inputs)

        self.assertEqual(model.input_channels, 7)
        self.assertEqual(outputs["heatmap"].shape, (1, 1, 32, 40))
        self.assertEqual(outputs["offset"].shape, (1, 2, 32, 40))
        self.assertEqual(outputs["size"].shape, (1, 2, 32, 40))
        self.assertEqual(outputs["rotation"].shape, (1, 2, 32, 40))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 3, 32, 40)))
        with self.assertRaises(ValueError):
            BEVDetector(input_channels=0)
        with self.assertRaises(TypeError):
            BEVDetector(input_channels=7.0)


if __name__ == "__main__":
    unittest.main()
